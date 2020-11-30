Gaze detection with crowdsourcing
=======
This project defines a framework for the detection of gazes in a crowdsourced environment. It was inspired by http://turkeyes.mit.edu.

##Setup
`pip install -e <PATH with setup.py>`

## Implementation on heroku
We use https://www.heroku.com/ to host the node.js implementation.

### Codecharts
todo

### Sentinel images
todo

The following royalty-free base images were used to create the sentinel images:
0. https://www.pexels.com/photo/women-s-white-and-black-button-up-collared-shirt-774909/
1. https://www.pexels.com/photo/woman-near-house-2804282/
2. https://www.pexels.com/photo/woman-standing-near-yellow-petaled-flower-2050994/
3. https://www.pexels.com/photo/person-holding-hammer-while-smiling-3267784/
4. https://www.pexels.com/photo/photography-of-laughing-guy-1408196/
5. https://www.pxfuel.com/en/free-photo-jmdxk
6. https://www.pexels.com/photo/man-in-blue-top-2830332/
7. https://www.pexels.com/photo/photo-of-man-wearing-denim-jacket-1599980/
8. https://www.pxfuel.com/en/free-photo-eibzf
9. https://www.pxfuel.com/en/free-photo-jrjqb

## Crowdsourcing job on appen
We use http://appen.com to run a crowdsourcing job. You need to create a cusotmer account to be able to create a launch crowdsourcing job.

### Cheaters
One of the disadvangtes of crowdsourcing is having to deal with workers that accept and do crowdsourcing jobs just for money (i.e., `cheaters`). The framework offers filtering mechanisms to remove data from such people from the dataset used for the analysis.

### Flagging cheaters
Cheaters can be reported by running `python /src/analysis/run_analysis.py` (required packages need to be installed). Running this script also rejects rows of data from cheaters in appen dats and triggers appen to acquire more data to replace the filtered rows.

## Analysis
Analysis can be started by running `python /src/qa/flag_cheaters.py` (required packages need to be installed).

### Visualisation
todo

### Config of analysis
Configuration of analysis needs to bde defined in `/src/analysis/config`. Please use the `config example` file for the required structure of the file.