# Riconoscimento Documenti Storici
Il progetto ha l'obiettivo di studiare i principali metodi per il riconoscimento di caratteri in documenti storici, oltre a sviluppare un approccio basato su tecniche di deep learning, al fine di riconoscere parole in foto di documenti storici potenzialmente danneggiati e/o con informazioni non deducibili dal contesto. Il processo di riconoscimento del testo richiede la risoluzione di diverse sfide, come la presenza di rumore, la distorsione e lo stato di conservazione delle immagini
## Passaggi principali
I tre step principali usati per raggiungere l'obbiettivo sono stati:
- Denoising delle immagini sorgente rappresentati i documenti storici
- Allenamento di modelli di Deep Learning
- Tecniche di OCR tramite modelli pre-allenati e modelli allenati dal gruppo di lavoro 
### Implementazione Denoising
Un processo di "pulizia" delle immagini originali è stato necessario al fine di ridurre il più possibile il "rumore" presenti in esse. Durante lo studio del progetto sono state prese in considerazione varie tecniche al fine di raggiungere questo scopo, tuttavia alcune si sono risultate inefficaci per il preciso scopo richiesto. 
Le tecniche che sono state lasciate nel programma finale sono perciò:
- Adaptive Thresholding: converte un'immagine in scala di grigi in un'immagine binaria, in cui i pixel sono classificati come bianchi o neri ma differenza del thresholding globale, in cui una soglia fissa viene applicata a tutta l'immagine, l'adaptive thresholding si adatta localmente alle variazioni locali di luminosità.
- Edge Detection: identifica le transizioni di contrasto o discontinuità nelle immagini, che spesso corrispondono a bordi o contorni degli oggetti.
Entrambe le tecniche sfruttano l'implementazione fornita nella libreria "CV2". 
### Implementazione OCR 
Le funzionalità di Optical Character Recgnition sono state necessarie al fine di poter ottenere una trascrizione delle foto rappresentanti i documenti storici di interesse. Durante lo studio si sono prese in considerazione le due librerie più sviluppate per questo scopo, ovvero "pytesseract" ed "EasyOCR", e si è potuto notare come utilizzando il modello di default fornito da entrambe le librerie il risultato fosse molto simile ma non soddisfacente. Si è perciò deciso di costruire un modello personalizzato focalizzandosi sulla libreria "EasyOCR", costruendo un dataset apoosito per poter permettere di allenare la Rete Neurale ad indentificare gli specifici documenti in questione. Si è ritenuto oppurtuno lasciare nell'implementazione finale la sola libreria "EasyOCR". 
## Funzionamento Script (./production)
Il prodotto finale può essere utilizzando da riga di comando tramite il comando eseguendo lo script contenuto nel file "main.py". 
Il formato degli argomenti sarà: "<input_dir> <output_dir> <denoising_option> <ocr_option>", dove:
- <input_dir>: percorso dove il programma andrà a prendere le immagini da processare (obbligatorio)
- <output_dir>: percorso dove il programma salverà il risultato della computazione sulle immagini in input, dividendole in cartelle (obbligatorio)
- <denoising_option>: metodo di denoising che il programma andrà ad applicare alle immagini in input (opzionale, al massimo una tecnica)
- <ocr_option>:  modello che la rete neurale utilizzerà per trascrivere i documenti in file .txt
### Opzioni di Denoising
I possibili metodi di denoising sono: 
- "-t" "--tresholding": applica la tecnica di Adaptive Thresholding alle immagini
- "-e" "--edgedetection": applica la tecnica di Edge Detection alle immagini
  
Il programma permette di applicare una sola tecnica per esecuzione.
### Modelli di OCR
Il programma permette all'utente di scegliere quale modello di allenamento della Rete Neurale utilizzare per il processo di OCR, tra le seguenti: 
- "-s" "--standardmodel": modello standard pre-allenato fornito dalla libreria "EasyOCR"
- "-m" "--machinewrittenmodel": modello ri-allenato specificamente per documenti storici scritti a macchina 
- "-w" "--handwrittenmodel": modello ri-allenato specificamente per documenti storici scritti a mano
