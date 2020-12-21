Gaze detection with crowdsourcing
=======
This project defines a framework for the detection of gazes in a crowdsourced environment. It was inspired by the [TurkEyes project](http://turkeyes.mit.edu). The [jsPsych framework](https://www.jspsych.org/) is used to for the frontend. Data is stored in a mongodb database. In the description below, it is assumed that that the repo is stored in the folder `gazes-crowdsourced`. Terminal commands lower assume macOS.

## Setup
Tested with Python 3.8.5. To setup the environment run these two commands:
- `pip install -e gazes-crowdsourced` will setup the project as a package accessible in the environment.
- `pip install -r gazes-crowdsourced/requirements.txt` will install required packages.

## Implementation on heroku
We use [Heroku](https://www.heroku.com/) to host the node.js implementation. The demo of the implementation may be viewed [here](https://gazes-crowdsourced.herokuapp.com/?debug=1&save_data=0). Implementation supports images and/or videos as stimuli.

### Codecharts
![example of codechart](https://github.com/bazilinskyy/gazes-crowdsourced/blob/master/public/img/codeboard/cb_0.jpg?raw=true)

Gaze information is gathered by showing a black screen with a set of random codes in the format 'LDD' (L=letter, D=digit, e.g., 'F32'). The participant is asked to remember the last code they were looking at. On the next page they need to input that code. Gazes are aggregated based on the mapping of codes to the coordinates within a stimulus. The mapping is available [here](https://github.com/bazilinskyy/gazes-crowdsourced/blob/master/public/img/codeboard/data.json).

### Sentinel images
![example of sentinel image](https://github.com/bazilinskyy/gazes-crowdsourced/blob/master/public/img/sentinel/sentinel_0.jpg?raw=true)

Sentinel images are oval images of faces. Participants are asked to look at them and input the code that is within the area of the face. Such images are used during training and during the study to see if people still pay attention to the task. We used the following royalty-free base images to create the sentinel images: [image 0](https://www.pexels.com/photo/women-s-white-and-black-button-up-collared-shirt-774909), [image 1](https://www.pexels.com/photo/woman-near-house-2804282), [image 2](https://www.pexels.com/photo/woman-standing-near-yellow-petaled-flower-2050994), [image 3](https://www.pexels.com/photo/person-holding-hammer-while-smiling-3267784), [image 4](https://www.pexels.com/photo/photography-of-laughing-guy-1408196), [image 5](https://www.pxfuel.com/en/free-photo-jmdxk), [image 6](https://www.pexels.com/photo/man-in-blue-top-2830332), [image 7](https://www.pexels.com/photo/photo-of-man-wearing-denim-jacket-1599980), [image 8](https://www.pxfuel.com/en/free-photo-eibzf), [image 9](https://www.pxfuel.com/en/free-photo-jrjqb).

### Training
Participants need to go through a training session before starting the experiment. They are allowed to fail the training a limited number of times. If they do not manage to finish the training session, they are not allowed to start the experiment. It is implemented by means of a cookie file.

The training session consists of 5 images of traffic scenes and 5 sentinel image. We used the following royalty-free images: [image 0](https://www.pxfuel.com/en/free-photo-xpoyf), [image 1](https://www.pxfuel.com/en/free-photo-xpoev), [image 2](https://www.pxfuel.com/en/free-photo-xnbzi), [image 3](https://www.pxfuel.com/en/free-photo-emhtx), [image 4](https://www.pxfuel.com/en/free-photo-ebgzh).

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
Configuration of analysis needs to be defined in `gazes-crowdsourced/gazes/analysis/config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. The config file has the following parameters:
* `appen_job`: ID of the appen job.
* `num_stimuli`: number of stimuli in the study.
* `stimulus_durations`: durations of showing each stimulus.
* `stimulus_width`: width of the stimulus.
* `stimulus_height`: height of the stimulus.
* `allowed_min_time`: the cut-off for minimal time of participation for filtering.
* `training_sent`: number of sentinel images during training.
* `allowed_mistakes_sent`: number of allowed mistakes with code input for sentinel images (excluding sentinel images in the training session).
* `allowed_cb_middle`: allowed ratio of coordinates of gazes within a square in the centre of the stimulus.
* `cb_middle_area`: width/2 and height/2 of the square in the middle of the stimulus for filtering based on coordinates.
* `mask_id`: number for masking of worker ID for anonymisation of data.
* `files_heroku`: files with data from heroku.
* `file_appen`: file with data from appen.
* `file_cheaters`: csv file with cheaters for flagging/.
* `mapping_cb`: json file with mapping of coordinates and codes in codeblocks for stimuli.
* `mapping_sentinel_cb`: json file with mapping of coordinates and codes in codeblocks for sentinel images.
* `path_stimuli`: path with stimuli.
