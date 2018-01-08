import Utils
import json
import subprocess
import os
import argparse

def extract_relationships(input_file, jar_path, descriptions_file, relations_file, categories_to_filter_out):
    all_text = []
    (jar_directory, filename) = os.path.split(jar_path);

    try:
        with open(input_file) as data_file:
            entitiesJsonList = json.loads(str(data_file.read()))
            for jsonEntity in entitiesJsonList:
                cleaned_text = Utils.getTextToProcess(Utils.Payload(jsonEntity), categories_to_filter_out)
                if cleaned_text:
                    all_text.append(cleaned_text)
    except IOError:
        print("Could not read file:" + input_file)

    try:
        with open(descriptions_file, 'w') as desc_file:
            for text in all_text:
                desc_file.write(text + "\n")
    except IOError:
        print("Could not read file:" + descriptions_file)

    print("Extracting relations using open IE")
    print("-----------------------------------")
    subprocess.call(["java", "-Xmx10g", "-XX:+UseConcMarkSweepGC", "-jar", jar_path, "--ignore-errors", "-s", "--format", "column", descriptions_file, relations_file], cwd=jar_directory)
    print("-----------------------------------")


if __name__ == '__main__':
    print("-----------------------------------")
    print("Extracting descriptions from input file")
    print("-----------------------------------")
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", help="Input Json file having the manual contents")
    argparser.add_argument("jar", help="Path of open ie jar")
    argparser.add_argument("categories_to_filter_out", help="string containing all the categories to filter out")
    argparser.add_argument("output_descriptions_file", help="Output file path for cleaned up text of all the entries in the manual.")
    argparser.add_argument("output_relations_file", help="Output file path for relationships extracted")
    args = argparser.parse_args()

    extract_relationships(args.input, args.jar, args.outputDesc, args.outputRel, args.categories_to_filter_out)

