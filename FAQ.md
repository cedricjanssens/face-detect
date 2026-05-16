# FAQ — face-detect

## Watch mode (FIFO daemon)

### Mon client hang au bout de N images, mais face-detect ne crash pas

**Cause la plus fréquente : le FIFO `out` n'est pas drainé assez vite côté client.**

Le pipe de sortie a un buffer kernel de 64 KB. Chaque réponse avec visages fait ~20-30 KB en JSON (embeddings 512d). Si le client (Node, Python, etc.) ne lit pas la réponse avant d'envoyer la requête suivante, 2-3 réponses non lues saturent le buffer → le `write()` du daemon bloque → deadlock.

**Diagnostic** :
```bash
# Si le daemon a logué "processing" mais pas "done" → c'est bien le write qui bloque
# Si "done" est logué mais le client ne reçoit rien → le read côté client est starved
```

**Solutions** :

1. **Toujours lire la réponse AVANT d'envoyer la requête suivante** (mode synchrone)
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

### Face-detect affiche "ANE skipped (ollama running)"

C'est normal et voulu. Ollama utilise MLX qui charge le Neural Engine. CoreML + MLX simultanés causent des deadlocks kernel irréversibles (processus en Uninterruptible state, survive même `kill -9`).

face-detect v0.5.2+ détecte Ollama au démarrage et force CPU+GPU. La différence de performance est minime (quelques dizaines de ms/image).

### Face-detect est bloqué et ne répond plus à `kill -9`

Vous avez un deadlock kernel du Neural Engine. Le processus est en UE (Uninterruptible state) :
```bash
ps aux | grep face-detect | awk '{print $8}'  # "U" = uninterruptible
```

**Seul remède : reboot.** Pour éviter à l'avenir :
- Ne jamais lancer face-detect + Ollama sans la protection ANE skip (v0.5.2+)
- Ne jamais lancer 2+ processus face-detect simultanés
- Variable de forçage : `FACE_DETECT_NO_ANE=1` (skip ANE même sans Ollama)

### Comment forcer CPU+GPU sans Ollama ?

```bash
FACE_DETECT_NO_ANE=1 face-detect photo.jpg
```

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
./helpers/face-detect/tests/run-tests.sh
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
