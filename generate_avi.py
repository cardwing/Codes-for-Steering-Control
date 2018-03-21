import cvbase as cvb

video = cvb.frames2video('/home/cardwing/Downloads/self-driving-car-master/steering-models/community-models/rambo/demo_final', '/home/cardwing/Downloads/self-driving-car-master/steering-models/community-models/rambo/demo_final_new.avi', fps = 20, filename_tmpl = '{:d}.png', start = 0,  end = 1998)
