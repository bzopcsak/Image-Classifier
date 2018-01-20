import tensorflow as tf, sys
import argparse
import os

def main():
    # Read in the image_data
    image_path = sys.argv[1]
    image_path2 = sys.argv[2]
    image_path3 = sys.argv[3]
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

        #score2 = predictions[0][0]
        #print(score2)

        #human_string2 = label_lines[0]
        #print(human_string2)
        #C:\Users\boldi\Desktop\ImageClassification\tensorflow\dif_images
        #os.rename(image_path, r"C:\Users\boldi\Desktop\ImageClassification\tensorflow\dif_images\star_wars\hey.jpg")
        #os.rename(image_path, "C:\Users\boldi\Desktop\ImageClassification\tensorflow\dif_images\star_wars")

if __name__ == "__main__":
   main()
