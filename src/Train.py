import TrainAndPredictUtils
import string, csv, argparse, pickle

def trainAndDumpModel(training_data, model_file):
    all_data=[]

    try:
        with open(training_data, 'r') as tsvin:
                tsvin = csv.reader(tsvin, delimiter='\t')
                for row in tsvin:
                    all_data.append((row[0], row[1]))
    except IOError:
        print("Could not read file:" + training_data)

    pipe = TrainAndPredictUtils.get_pipeline()
    X = [x[0] for x in all_data]
    Y = [x[1] for x in all_data]

    pipe.fit(X, Y)

    try:
        with open(model_file, 'wb') as fid:
            pickle.dump(pipe, fid)
    except IOError:
        print("Could not read file:" + model_file)

if __name__ == '__main__':
    print("-----------------------------------")
    print("Training model from training data")
    print("-----------------------------------")
    argparser = argparse.ArgumentParser()
    argparser.add_argument("training_data", help="Training data with positive and negative examples")
    argparser.add_argument("model_file", help="Path of model file output")
    args = argparser.parse_args()

    trainAndDumpModel(args.training_data, args.model_file)
    try:
        with open(args.model_file, 'rb') as fid:
            model = pickle.load(fid)
    except IOError:
        print("Could not read file:" + args.model_file)

    score = model.score(X, Y)
    print(score)