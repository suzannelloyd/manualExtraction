# ManualExtraction
Project to extract from a vehicle manual

# Prerequisites
This project uses Python3.5. Using [anaconda](https://conda.io/docs/user-guide/install/index.html), you can create a new environment with this command:

    conda create -n myenv python=3.5

The project uses the following prerequisites that can be installed via pip or conda. Assuming python and pip to be already installed.
    
    pip install -U spacy
    pip install numpy
    pip install scipy
    pip install sklearn
    pip install -U nltk

 We use google cloud api to get entities as well as spacy's noun chunks api. For google api, please go through [Google API Getting Started](https://cloud.google.com/natural-language/docs/quickstart) to establish your credentials and set google application credentials path locally. Install python client library
 
    pip install --upgrade google-api-python-client

We use [OpenIE](http://openie.allenai.org/) for relation extraction. Download openie jar from [here](https://drive.google.com/open?id=1Ip3Jko-jY97EIStYl4LEeuQHZfwitYdm) to the local [OpenIE](https://github.com/Srivatsava/manualExtraction/tree/master/OpenIE) folder.

In the same folder, download the standalone jar for BONIE from [here](https://github.com/dair-iitd/OpenIE-standalone/releases/download/v5.0/BONIE.jar) and place it inside a `lib` folder(create the `lib` folder parallel to the `src` folder).

Also, download the standalone jar for Conjunctive Sentences work from [here](https://github.com/dair-iitd/OpenIE-standalone/releases/download/v5.0/ListExtractor.jar) and place it inside the `lib` folder.

Extractions from Conjunctive Sentences uses Berkeley Language Model. Download the Language Model file from [here](https://drive.google.com/file/d/0B-5EkZMOlIt2cFdjYUJZdGxSREU/view?usp=sharing) and place it inside a data folder(create the `data` folder parallel to the `src` folder)

# Running the code

To run the entire code use the following command:

    python ExtractManual.py <path_to_input_file> <path_to_openie_jar> <path_to_output_folder> <path_to_model>

path_to_input_file: This is the path to input json file included in the [Input](https://github.com/Srivatsava/manualExtraction/tree/master/Input) folder.
path_to_openie_jar: This is the path to openie jar file included in the [OpenIE](https://github.com/Srivatsava/manualExtraction/tree/master/OpenIE) folder.
path_to_output_folder: This is the path to openie output folder where we want the output files to be in.
path_to_model: This is the path to model (.pkl) file included in the [Model](https://github.com/Srivatsava/manualExtraction/tree/master/Model) folder. 

This will automatically extract the relationships using openie, entities using spacy and google api, applies the model to improve precision on the extracted entities and filters out relationships that have a subject and object that are not related to the extracted entities.

There are two more utils you can use to run.
Train Model:

    python Train.py <path_to_training_data_file> <path_to_model>

path_to_training_data_file: This is the path to training data file included in the [Model](https://github.com/Srivatsava/manualExtraction/tree/master/Model) folder. 
path_to_model: This is the path to output model file where we want to put our model. 
This command trains the linear svm model to train on the training dataset and dump the model. It also reports training accuracy.

Extract relationships:
    
    python ExtractRelationships.py <path_to_input_file> <path_to_openie_jar> -outputDesc <path_to_output_descriptions> <path_to_output_relationships>
    
path_to_input_file: This is the path to input json file included in the [Input](https://github.com/Srivatsava/manualExtraction/tree/master/Input) folder.
path_to_openie_jar: This is the path to openie jar file included in the [OpenIE](https://github.com/Srivatsava/manualExtraction/tree/master/OpenIE) folder.
path_to_output_descriptions: This is the path to the output cleaned descriptions file.
path_to_output_relationships: This is the path to the output relationships file.

To test on a golden set using extracted relationships, you can use:

    python ExtractManual.py <path_to_input_file> <path_to_openie_jar> <path_to_output_folder> <path_to_model> --extractedRel <path_to_extracted_relationships> --goldenset <path_to_goldenset>
    
path_to_goldenset: The true entities that we want to extract. We have included a sample file [here](https://github.com/Srivatsava/manualExtraction/blob/master/Input/goldenset_child.tsv).

We also included a test model file that tests different models on training data. To run that:

    python ModelTester.py
    
 Make sure to replace the input training_data file in this method before running it.
 
 # Output Files
 A number of sample output files were included for evaluation purposes in the [OutputSample](https://github.com/Srivatsava/manualExtraction/tree/master/OutputSample) Folder. 
 
 These are the results of running the above code on a [Small dataset](https://github.com/Srivatsava/manualExtraction/blob/master/Input/output_child.json).
 
[Stats](https://github.com/Srivatsava/manualExtraction/blob/master/OutputSample/stats.txt) contains all the stats of running the code on this input file before and after cleanup and after applying filtering on this [file](https://github.com/Srivatsava/manualExtraction/blob/master/Input/output_child.json)

[ModelComparisonStats](https://github.com/Srivatsava/manualExtraction/blob/master/OutputSample/modelComparison.txt) contains all the stats of different models tried on the training dataset for labeling positive and negative entity examples [trainingFile](https://github.com/Srivatsava/manualExtraction/blob/master/Model/trainingdata.tsv)

[Cooccuring Google API Entities](https://github.com/Srivatsava/manualExtraction/blob/master/OutputSample/output_coge_samesentence_child.csv): Entities that are discovered using Google API that cooccur together.

[Cooccuring Spacy NounChunks Entities](https://github.com/Srivatsava/manualExtraction/blob/master/OutputSample/output_conounchunks_samesentence_child.csv): Entities that are discovered using Spacy Nounchunks that cooccur together.

[Google API Entities](https://github.com/Srivatsava/manualExtraction/blob/master/OutputSample/output_ge_child.csv): Entities that are discovered using Google API and filtered using our ranker.

[Spacy NounChunks Entities](https://github.com/Srivatsava/manualExtraction/blob/master/OutputSample/output_nounchunks_child.csv): Entities that are discovered using Spacy Nounchunks and filtered using our ranker.

[Google API Relations](https://github.com/Srivatsava/manualExtraction/blob/master/OutputSample/output_rel_ge_child.csv): Relations extracted from openIE and Google API entities.

[SpacyNounChunks API Relations](https://github.com/Srivatsava/manualExtraction/blob/master/OutputSample/output_rel_nc_child.csv): Relations extracted from openIE and Spacy Nounchunk entities.
 
Final Results on the goldenset for this small dataset:

    Noun chunk stats (precision, recall, fscore): 0.93,0.74,0.82
    Google API Entity stats (precision, recall, fscore): 0.97,0.57,0.72
 
This also extracted some really good entities and relations as can be seen [here](https://github.com/Srivatsava/manualExtraction/blob/master/OutputSample/output_nounchunks_child.csv) and [here](https://github.com/Srivatsava/manualExtraction/blob/master/OutputSample/output_rel_nc_child.csv)


 
 
 




