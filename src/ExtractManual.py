import argparse, json, os
import ExtractRelationships
import Utils

#These categories are either not formated well or not extracted well for us to extract entities or relations.
categories_to_filter_out = 'No Category;Contents;Reference;Technical data;Everything from A to Z'

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", help="Input Json file having the manual contents")
    argparser.add_argument("openiejar", help="Path of open ie jar used to extract relationships")
    argparser.add_argument("output_dir", help="Output Directory containing all the output files")
    argparser.add_argument("model", help="Model file for filtering extracted entities")
    argparser.add_argument("--extractedRel", help="(optional) Extracted relationships file. If not specified, we auto discover them.", required=False)
    argparser.add_argument("--goldenset", help="(optional) Goldenset file to compute precision and recall", required=False)
    args = argparser.parse_args()

    if args.extractedRel:
        relations_file = args.extractedRel
    else:
        descriptions_file = os.path.join(args.output_dir, "descriptions.tsv")
        relations_file = os.path.join(args.output_dir, "relations.tsv")
        ExtractRelationships.extract_relationships(args.input, args.openiejar, descriptions_file, relations_file, categories_to_filter_out)

    #Process data once and get all the entities and information needed for further analysis
    try:
        with open(args.input) as data_file:
            data = json.loads(str(data_file.read()))
            processed_data, cooccuring_cnt_dicts, counts_dict = Utils.processData(data, categories_to_filter_out, args.model)
    except IOError:
        print("Could not read file:" + args.input)

    relations_dict = Utils.filterRelations(relations_file, counts_dict)
    print()

    if args.goldenset:
        Utils.CalculateAndWriteMetrics(args.goldenset, counts_dict, args.output_dir)

    Utils.SortAndWriteEntities(counts_dict, args.output_dir)
    Utils.SortAndWriteEntities(cooccuring_cnt_dicts, args.output_dir, True)