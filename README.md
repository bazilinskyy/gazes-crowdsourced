Gaze detection with crowdsourcing
=======
This project defines a framework for the detection of gazes in a crowdsourced environment. It was inspired by the [TurkEyes project](http://turkeyes.mit.edu). The [jsPsych framework](https://www.jspsych.org/) is used to for the frontend. Data is stored in a mongodb database. In the description below, it is assumed that that the repo is stored in the folder `gazes-crowdsourced`. Commands lower assume macOS.

## Setup
To setup environment run these two commands:
- `pip install -e gazes-crowdsourced` will setup the project as a package accessible in the environment.
- `pip install -r gazes-crowdsourced/requirements.txt` will install required packages.

## Implementation on heroku
We use [Heroku](https://www.heroku.com/) to host the node.js implementation. The demo of the implementation may be viewed [here](https://gazes-crowdsourced.herokuapp.com/?debug=1&save_data=0). Implementation supports images and/or videos as stimuli.

### Codecharts
![example of codechart](https://github.com/bazilinskyy/gazes-crowdsourced/blob/master/public/img/codeboard/cb_0.jpg?raw=true)

Gaze information is gathered by showing a black screen with a set of random codes in the format 'LDD' (L=letter, D=digit, e.g., 'F32'). The participant is asked to remember the last code they were looking at. On the next page they need to input that code. Gazes are aggregated based on the mapping of codes to the coordinates within a stimulus. The mapping is available [here](https://github.com/bazilinskyy/gazes-crowdsourced/blob/master/public/img/codeboard/data.json).

### Sentinel images
![example of sentinel image](https://github.com/bazilinskyy/gazes-crowdsourced/blob/master/public/img/sentinel/sentinel_0.jpg?raw=true)

Sentinel images are oval images of faces. Participants are asked to look at them and input the code that is within the area of the face. Such images are used during training and during the study to see if people still pay attention to the task.

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

### Training
Participants need to go through a training session before starting the experiment. They are allowed to fail the training a limited number of times. If they do not manage to finish the training session, they are not allowed to start the experiment. It is implemented by means of a cookie file.

### Filtering of heroku data
Data from heroku is filtered based on the following criteria:
1. People who entered incorrect codes for sentinel images more than `config.allowed_mistakes_sent` times.

## Crowdsourcing job on appen
We use [appen](http://appen.com) to run a crowdsourcing job. You need to create a client account to be able to create a launch crowdsourcing job. Preview of the appen job used in this experiment is available [here](https://view.appen.io/channels/cf_internal/jobs/1670895/editor_preview?token=ne2tN-bKLMxl-YCvOGV-YA).

### Filtering of appen data
Data from appen is filtered based on the following criteria:
1. People who did not read instructions.
2. People that are under 18 years of age.
3. People who completed the study in under `config.allowed_min_time` min.
4. People who completed the study from the same IP more than once (the 1st data entry is retained).
5. People who used the same `worker_code` multiple times. One of the disadvantages of crowdsourcing is having to deal with workers that accept and do crowdsourcing jobs just for money (i.e., `cheaters`). The framework offers filtering mechanisms to remove data from such people from the dataset used for the analysis. Cheaters can be reported from the `gazes.qa.QA` class. It also rejects rows of data from cheaters in appen data and triggers appen to acquire more data to replace the filtered rows.

### Anonymisation of data
Data from appen is anonymised in the following way:
1. IP addresses are assigned to a mask starting from `0.0.0.0` and incrementing by 1 for each unique IP address (e.g., the 257th IP address would be masked as `0.0.0.256`).
2. IDs are anonymised by subtracting the given ID from `config.mask_id`.

## Analysis
Analysis can be started by running `python gazes-crowdsourced/gazes/analysis/run_analysis.py`. A number of csv files are saved in `gazes-crowdsourced/_output`.

### Visualisation
Heatmaps and visualisations of gazes are saved in `gazes-crowdsourced/_output`. 
todo: add example image with heatmap

### Configuration of analysis
Configuration of analysis needs to be defined in `gazes-crowdsourced/gazes/analysis/config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. 
