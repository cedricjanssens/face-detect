# FAQ — face-detect

## Watch mode (FIFO daemon)

### Format des embeddings (mitigation FIFO macOS 16 KB)

Le buffer kernel d'une FIFO macOS commence à **16 KB** (`PIPE_SIZE`) et peut grandir dynamiquement à 64 KB mais redescend après drain. Une réponse `face-detect` avec 3-4 visages dépasse facilement 16 KB → le `write()` du daemon bloque, le client voit du JSON tronqué.

face-detect v0.5.5+ propose deux mitigations :

1. **Précision réduite (défaut)** : embeddings sérialisés à 6 chiffres significatifs au lieu du round-trip IEEE complet (~10 chiffres). Erreur cosine < 1e-6 → invisible pour la reconnaissance. Réduit la taille de ~10%.

2. **Format base64** : flag `--embedding-format b64` → le champ `embedding` (array de 512 floats) devient `embedding_b64` (string base64 de 2732 chars). Réduit la taille de ~30%.

Décodage Node :
```js
const buf = Buffer.from(face.embedding_b64, 'base64')
const embedding = new Float32Array(buf.buffer, buf.byteOffset, 512)
```

⚠️ Même avec b64, une image avec 5+ visages peut encore saturer le buffer FIFO. Le fix définitif (transport Unix socket) arrive en 0.6.0.

### Contrat client : 1 requête in-flight maximum

**Règle stricte : envoyer une requête dans `in`, lire la réponse complète sur `out`, PUIS envoyer la suivante.**

Le daemon est single-threaded par design. Plusieurs requêtes en parallèle dans la FIFO `in` ne sont **pas** "absorbées" plus vite — elles saturent le buffer kernel de sortie (64 KB) en quelques réponses, le `write()` du daemon bloque, et les réponses qui *arrivent* à passer s'entremêlent → JSON corrompu côté client.

**Anti-patterns observés** :
- ❌ Pool de N workers qui pushent en parallèle dans la même FIFO `in`
- ❌ "Queue avec _maxInflight=4" qui injecte 4 requêtes avant d'attendre les réponses
- ❌ Fire-and-forget ("j'enverrai la suivante quand j'y penserai")

Si vous voulez du parallélisme côté API JS/Python, gardez la queue côté client mais sérialisez à **1 in-flight strict** au niveau de l'écriture FIFO.

### Mon client hang au bout de N images, mais face-detect ne crash pas

**Cause la plus fréquente : violation du contrat 1-in-flight ci-dessus, ou FIFO `out` pas drainé assez vite.**

Chaque réponse avec visages fait ~20-30 KB en JSON (embeddings 512d). 2-3 réponses non lues = buffer saturé = deadlock.

**Diagnostic** :
```bash
# Si le daemon a logué "processing" mais pas "done" → c'est bien le write qui bloque
# Si "done" est logué mais le client ne reçoit rien → le read côté client est starved
# Si "Expected ',' or ']' after array element" côté client → réponses entremêlées, plusieurs requêtes étaient in-flight
```

**Solutions** :

1. **`_maxInflight = 1` strict** au niveau de l'écriture FIFO (cf. section précédente)
2. **Reader dédié haute priorité** — un stream reader séparé du event loop CPU-bound :
   ```javascript
   // Node.js — reader sur un fd séparé, pas bloqué par le processing
   const outStream = fs.createReadStream(fifoOutPath, { encoding: 'utf8' });
   const rl = readline.createInterface({ input: outStream });
   rl.on('line', (line) => { /* parse JSON, resolve pending promise */ });
   ```
3. **Ne jamais fire-and-forget** — chaque write dans `in` doit avoir un read correspondant sur `out`

### Face-detect est lent (>500ms par image)

Temps normaux sur Apple Silicon :
- **Image simple (1 visage)** : 80-250ms
- **Image multi-visages (3+)** : 150-400ms (1 embedding par visage)
- **Grosse image (>8 Mpx)** : 200-500ms (décodage + détection)
- **Première image** : +100-200ms (warm-up Vision framework)

Si c'est beaucoup plus lent, vérifier :
- Le modèle est bien sur le disque local (pas un NAS ou volume réseau)
- Pas de swap mémoire (`vm_stat | grep "Pageouts"`)

### Puis-je lancer plusieurs daemons face-detect en parallèle ?

**Non.** Un seul daemon à la fois. Plusieurs processus CoreML concurrents se battent pour le Neural Engine et peuvent provoquer un deadlock kernel (UE state irréversible, seul un reboot le résout).

Utilisez le watch mode comme multiplexeur : un seul daemon sert tous les clients séquentiellement.

---

## Neural Engine et Ollama

### Face-detect affiche "ANE skipped (Ollama MLX runner detected)"

Seul un runner **MLX** d'Ollama charge le Neural Engine. CoreML + MLX simultanés causent des deadlocks kernel irréversibles (processus en Uninterruptible state, survive même `kill -9`).

face-detect v0.5.4+ détecte spécifiquement les runners MLX (`pgrep -f 'ollama.runner.*mlx'`). Si seul `ollama serve` tourne, ou si le runner actif est `--ollama-engine` (llama.cpp + Metal GPU, ex: `nomic-embed-text`, modèles GGUF), l'ANE reste activé.

**Avant v0.5.4** : tout processus Ollama déclenchait le skip, même un simple embedding model GPU. Comportement trop conservateur, corrigé.

### Face-detect est bloqué et ne répond plus à `kill -9`

Vous avez un deadlock kernel du Neural Engine. Le processus est en UE (Uninterruptible state) :
```bash
ps aux | grep face-detect | awk '{print $8}'  # "U" = uninterruptible
```

**Seul remède : reboot.** Pour éviter à l'avenir :
- Ne jamais lancer face-detect + Ollama-MLX sans la protection ANE skip (v0.5.4+)
- Ne jamais lancer 2+ processus face-detect simultanés
- Variable de forçage : `FACE_DETECT_NO_ANE=1` (skip ANE inconditionnel)

### Predict watchdog (mode watch, v0.5.4+)

face-detect arme un watchdog par requête en mode `--watch`. Si un seul `processImage()` dépasse `--predict-timeout` secondes (60s par défaut), le daemon **exit 124** avec en stderr :

```
face-detect: predict watchdog fired (>predict-timeout, likely ANE/CoreML deadlock), exiting 124
```

C'est un signal pour le client : « j'étais coincé sur une image, je sors pour que tu puisses me relancer ». **Important** : ne **pas** retry la même image sur le daemon respawné — elle a sûrement déclenché le deadlock. Skip et passe à la suivante.

Cas limites :
- Si le predict est en **UE complète**, `_exit(124)` peut lui-même bloquer. Le client doit donc *aussi* avoir un heartbeat timeout (ping/pong) → respawn brutal. Le watchdog est une protection best-effort qui couvre les cas "lent mais pas mort".
- À ajuster avec `--predict-timeout 30` si vos images sont petites et le throughput compte, ou `--predict-timeout 120` pour des batches photos lourdes.

### Comment forcer CPU+GPU ?

```bash
FACE_DETECT_NO_ANE=1 face-detect photo.jpg     # toujours skip ANE
```

### Comment forcer l'ANE malgré la détection MLX ?

```bash
FACE_DETECT_FORCE_ANE=1 face-detect photo.jpg  # bypass la détection (à vos risques)
```

À n'utiliser que si vous savez que le runner MLX détecté ne touche pas l'ANE (cas rare). Risque de deadlock kernel sinon.

---

## Intégration

### Quelle taille fait une réponse JSON ?

| Visages | Taille approximative |
|---------|---------------------|
| 0       | ~500 bytes          |
| 1       | ~15-20 KB           |
| 2       | ~30-35 KB           |
| 3       | ~45-50 KB           |
| N       | ~15 KB × N          |

Les embeddings 512d en JSON (texte flottant) sont le gros du payload. Si vous parsez beaucoup de réponses, prenez en compte cette taille pour le sizing de vos buffers.

### Le champ `id` dans le protocole watch

Tout champ `"id"` envoyé dans la requête est propagé tel quel dans la réponse. C'est le mécanisme de corrélation pour les clients async :
```json
{"id":"batch-42","image":"/path/to/img.jpg"}
→ {"id":"batch-42","image":"/path/to/img.jpg","faces":[...],...}
```

### Ping et health check

```json
{"ping":true,"id":"hb-1"}
→ {"pong":true,"id":"hb-1","uptime_ms":45000,"processed":12,"engine":"adaface","engine_dim":512,"model":"ir18"}
```

Utilisez ping pour vérifier que le daemon est vivant sans traiter d'image.

### Shutdown propre

```json
{"shutdown":true,"id":"bye"}
→ {"shutdown":true,"id":"bye","uptime_ms":120000,"processed":87}
```

Le daemon écrit la réponse, flush (`synchronizeFile`), et fait `_exit(0)`. Pas besoin de SIGTERM après un shutdown JSON.

### Que se passe-t-il si le daemon reçoit du JSON invalide ?

Il log une erreur sur stderr (`malformed JSON`) et **ignore la requête** — pas de réponse envoyée. Le client en attente d'une réponse sera bloqué indéfiniment. Validez votre JSON avant de l'envoyer.

---

## Images et formats

### Formats supportés

HEIC, JPEG, PNG, TIFF — via macOS ImageIO (CGImageSource). Pas de RAW, pas de WebP.

### Images éditées par Picasa (dual JFIF+Exif header)

Face-detect traite correctement les images retouchées par Picasa (header JFIF + Exif combinés). Aucun problème connu avec ces fichiers.

### Orientation EXIF

L'orientation est gérée par CGImageSource au chargement. Les embeddings sont calculés sur l'image correctement orientée.

### Image sans visage

Retourne `"faces": []` (tableau vide, pas null), `"description": "image"` comme fallback.

---

## Tests

### Lancer la suite de non-régression

```bash
make test
# or directly:
./tests/run-tests.sh
```

45 assertions, 11 groupes de tests. Tolère un daemon face-detect pré-existant (archiviste) sans faux positif au test "zero zombies".

### Le test échoue avec "CLI disabled"

Les modes CLI (single, batch, video, bench) nécessitent `FACE_DETECT_ALLOW_CLI=1` en variable d'environnement. Le script de test le set automatiquement. Si vous testez manuellement :
```bash
FACE_DETECT_ALLOW_CLI=1 face-detect photo.jpg
```

---

## Modèles

### IR-18 vs IR-50 — lequel choisir ?

| | IR-18 (default) | IR-50 |
|---|---|---|
| Taille | 46 MB | 83 MB |
| Vitesse | ~80-100ms/face | ~120-150ms/face |
| Précision | Bonne | Meilleure discrimination |
| Usage | Screening rapide | Clustering fin, validation |

Pour le traitement de masse (archiviste), IR-18 est le bon choix. IR-50 est utile pour résoudre des cas ambigus (même personne à différents âges, jumeaux).

```bash
face-detect --model ir50 photo.jpg
```

### Où sont les modèles ?

```bash
ls /opt/homebrew/share/face-detect/
# AdaFace_IR18.mlpackage/
# AdaFace_IR50.mlpackage/
```
