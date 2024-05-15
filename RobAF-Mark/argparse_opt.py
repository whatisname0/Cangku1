import argparse
import math

def arg():
	parser = argparse.ArgumentParser()
	parser.add_argument('--R', type=float, default=40*math.sqrt(2), help='the radius of the select circle')
	parser.add_argument('--size_block', type=int, default=32, help='the w and h of inserted block') 
	parser.add_argument('--delta', type=str, default=8, help='特征点之间的位置误差')
	parser.add_argument('--mode', type=str, default=False, help='the way to attack the img')  

	parser.add_argument('--if_insert', type=bool, default=True, help='mode of insert')
	parser.add_argument('--if_extract', type=bool, default=True, help='mode of extract')
	parser.add_argument('--bite_num', type=int, default=4, help='select_num_of_bite') 

	parser.add_argument('--weights_path', type=str, default='./model/superpoint/superpoint_v1.pth',
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
	parser.add_argument('--img_glob', type=str, default='*.jpg',
      help='Glob match if directory of images is specified (default: \'*.png\').') 
	parser.add_argument('--skip', type=int, default=1,
      help='Images to skip if input is movie or directory (default: 1).')
	parser.add_argument('--show_extra', action='store_true',
      help='Show extra debug outputs (default: False).')
	parser.add_argument('--H', type=int, default=120,
      help='Input image height (default: 120).')
	parser.add_argument('--W', type=int, default=160,
      help='Input image width (default:160).')
	parser.add_argument('--display_scale', type=int, default=2,
      help='Factor to scale output visualization (default: 2).')
	parser.add_argument('--min_length', type=int, default=2,
      help='Minimum length of point tracks (default: 2).')
	parser.add_argument('--max_length', type=int, default=5,
      help='Maximum length of point tracks (default: 5).')
	parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) distance (default: 4).')
	parser.add_argument('--conf_thresh', type=float, default=0.015,
      help='Detector confidence threshold (default: 0.015).')
	parser.add_argument('--nn_thresh', type=float, default=0.7,
      help='Descriptor matching threshold (default: 0.7).')
	parser.add_argument('--camid', type=int, default=0,
      help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
	parser.add_argument('--waitkey', type=int, default=1,
      help='OpenCV waitkey time in ms (default: 1).')
	parser.add_argument('--cuda', action='store_true',
      help='Use cuda GPU to speed up network processing speed (default: False)')
	parser.add_argument('--no_display', action='store_true',
      help='Do not display images to screen. Useful if running remotely (default: False).')
	parser.add_argument('--write', action='store_true',default=True,  
      help='Save output frames to a directory (default: False)')  

	args = parser.parse_args()
	return args

