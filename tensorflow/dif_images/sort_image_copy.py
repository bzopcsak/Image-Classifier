import tensorflow as tf, sys
import argparse
import os
import glob
from shutil import copyfile

def callFirst():
    image_path2 = sys.argv[2]
    image_path3 = sys.argv[3]
    destination = sys.argv[4]
    
    path = sys.argv[1]
    for filename in glob.glob(os.path.join(path, '*.jpg')):
        main(filename,image_path2,image_path3,destination)

    for filename in glob.glob(os.path.join(path, '*.png')):
        main(filename,image_path2,image_path3,destination)

    for filename in glob.glob(os.path.join(path, '*.gif')):
        main(filename,image_path2,image_path3,destination)
        

def main(image_path,image_path2,image_path3,destination):
    # Read in the image_data
    
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in open(image_path2)]

    # Unpersists graph from file
    with open(image_path3, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s: %.1f' % (human_string, (round(score,3)*100)) + "%")
            
            
        print("\n")

        #score2 = predictions[0][0]
        #print(score2)

        #human_string2 = label_lines[0]
        i = 0
        for node_id in top_k:
            if(i == 0):
                human_string2 = label_lines[node_id]
            i = i+1
        human_string2WithNoSpace = human_string2.replace(" ","-")
        #print(human_string2)


        backslash = "\\"


        path = destination + backslash + human_string2WithNoSpace
        if not os.path.exists(path):
            os.makedirs(path)


        fileName = image_path.split(backslash)[-1]

        toRename = path + backslash + fileName

        #print(image_path)
        #print(toRename)

        copyfile(image_path, toRename)
        

if __name__ == "__main__":
   callFirst()
