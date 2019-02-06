import demo_image 
import argparse
import cv2
import time
import config
from model import get_testing_model
import json
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--output', type=str, default='result.png', help='output image')
	parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
	args = parser.parse_args()
	output = args.output
	keras_weights_file = args.model

	tic = time.time()
	print('start processing...')
	# load model
	# authors of original model don't use
	# vgg normalization (subtracting mean) on input images
	model = get_testing_model(38,19)
	model.load_weights(keras_weights_file)
	# load config
	_config = config.GetConfig("Canonical")
	toc = time.time()
	print ('processing time is %.5f' % (toc - tic))
	cam = cv2.VideoCapture(0)
	frame = 0
	while True:
		ret_val, canvas = cam.read()
		if ret_val:
			scanvas, pose_keypoints = demo_image.process(canvas,
														_config,
														model)

		with open(os.path.join("./json", '{0}_keypoints.json'.format(str(frame).zfill(12))), 'w') as outfile:
			 json.dump(pose_keypoints, outfile)

		cv2.imshow('keras Pose Estimation', canvas)
		cv2.imwrite(output, canvas)
		frame += 1
		if cv2.waitKey(1) == 27:
			break

	cv2.destroyAllWindows()