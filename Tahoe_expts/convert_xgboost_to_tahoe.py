import json
import math
import xgboost as xgb
import os
import tempfile
import csv
import numpy
import pandas

class ConvertToTahoe:
    def __init__(self, model_file_path, output_dir):
        self.model_file_path = model_file_path
        self.data_file_path = model_file_path + ".test.sampled.csv"

        self.output_file_path = os.path.join(output_dir, os.path.basename(model_file_path) + ".txt")
        self.output_file_name = os.path.basename(model_file_path) + ".txt.test.sampled.txt"
        self.set_model_parameters()

        temp_dir = tempfile.gettempdir()
        self.dumped_model_path = os.path.join(temp_dir, os.path.basename(model_file_path) + "_temp.json.txt")
        booster = xgb.Booster()
        booster.load_model(model_file_path)
        booster.dump_model(self.dumped_model_path, with_stats=True, dump_format='json')

    def __del__(self):
        os.remove(self.dumped_model_path)

    def get_tree_depth(self, num_nodes):
        return math.ceil(math.log(num_nodes+1,2))
    
    def set_model_parameters(self):
        with open(self.model_file_path) as myjson:
            data = myjson.read()

        obj = json.loads(data)
        max_depth = 0
        for tree in obj['learner']['gradient_booster']['model']['trees']:
            num_nodes = int(tree['tree_param']['num_nodes'])
            max_depth = max(max_depth, self.get_tree_depth(num_nodes))

        self.depth = max_depth
        self.trees = int(obj['learner']['gradient_booster']['model']['gbtree_model_param']['num_trees'])
        self.isclassifier = int(obj['learner']['learner_model_param']['num_class']) > 1
        self.feature_names = obj['learner']['feature_names']
    
    def construct_inputs(self, csvPath : str, batch_size) -> numpy.ndarray:
        data_df = pandas.read_csv(csvPath, header=None)
        full_test_array = numpy.array(data_df, order='C')
        num_repeats = 1
        if (full_test_array.shape[0] < batch_size):
            num_repeats = math.ceil(batch_size/full_test_array.shape[0])
        full_test_array = numpy.tile(full_test_array, (num_repeats, 1))
        inputs = numpy.array(full_test_array[0:batch_size, :-1], numpy.float32, order='C')
        return inputs

    def convert_data_file(self, batch_size):
        inputs = self.construct_inputs(self.data_file_path, batch_size)
        
        # Create the output directory if it doesn't exist
        
        output_dir = os.path.join(os.path.dirname(self.output_file_path), str(batch_size))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file_path = os.path.join(output_dir, self.output_file_name)

        with open(output_file_path, 'w') as tahoe_data_file:
            tahoe_data_file.writelines(str(len(inputs))) # Number of rows
            tahoe_data_file.writelines("\n")
            tahoe_data_file.writelines(str(len(inputs[0]))) # Number of columns
            tahoe_data_file.writelines("\n")
            tahoe_data_file.writelines(str(0.0))
            tahoe_data_file.writelines("\n")
            for row in inputs:
                for col in row:
                    tahoe_data_file.writelines(str(col))
                    tahoe_data_file.writelines("\n")

    def convert(self):
        with open(self.dumped_model_path) as myjson:
            data = myjson.read()

        obj = json.loads(data)

        depth_tree = self.depth+1
        num_node = pow(2,depth_tree) - 1
        num_tree = self.trees
        values = [None] * num_tree
        for i in range(num_tree):
            values[i] = [None] * num_node
            for j in range(num_node):
                values[i][j] = [None] * 5


        for i in range(num_tree):
            for j in range(num_node):
                root = obj[i]
                depth = math.ceil(math.log(j+2,2)) - 1
                index = j+1
                mystack = []
                for _ in range(depth):
                    mystack.append(index%2)
                    index = int(index/2)
                for _ in range(len(mystack)):
                    if 'children' in root:
                        if mystack.pop() == 0:
                            root = root['children'][0]
                        else:
                            root = root['children'][1]
                    else:
                        root = None
                        break
                if root is not None:
                    if 'children' in root:
                        values[i][j][0] = root['split']
                        values[i][j][1] = root['split_condition']
                        values[i][j][2] = root['missing']%2
                        values[i][j][3] = root['cover']
                        values[i][j][4] = 0
                    else:
                        values[i][j][0] = None
                        values[i][j][1] = root['leaf']
                        values[i][j][2] = None
                        values[i][j][3] = root['cover']
                        values[i][j][4] = 1

        with open(self.output_file_path, "w") as f:
            f.writelines(str(num_tree))
            f.writelines('\n')
            f.writelines(str(depth_tree))
            f.writelines('\n')
            for i in range(num_tree):
                for j in range(num_node):
                    for k in range(5):
                        if k == 0 and values[i][j][k]:
                            values[i][j][k] = self.feature_names.index(values[i][j][k])
                        f.writelines(str(values[i][j][k]))
                        f.writelines('\n')

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert xgboost model to tahoe model')
    parser.add_argument('--model_dir', type=str, help='Path to the directory with xgboost model(s).')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory.')
    args = parser.parse_args()

    model_files = [os.path.join(args.model_dir, file) for file in os.listdir(args.model_dir) if file.endswith('save.json')]
    for model_file in model_files:
        converter = ConvertToTahoe(model_file, args.output_dir)
        print("Converting model", model_file)
        converter.convert()
        print("Finished converting", model_file)
        batch_sizes = [512, 1024, 2048, 4096, 8192, 16384]
        for batch_size in batch_sizes:
            print("Converting data file for model", model_file, "with batch size", batch_size)
            converter.convert_data_file(batch_size)
            print("Finished converting", model_file, "with batch size", batch_size)
