import cv2
import numpy as np
import glob
from operator import itemgetter
import csv
import os

dataset = []

def dataset_init(dirname, verbose=True):
    
    # Load dataset from folders (training)
    files = os.listdir(dirname)
    n_keypoints = []

    # Initiate SURf detector and set the Hessian Threshold like explain in:
    # https://stackoverflow.com/questions/18744051/opencv-surf-hessian-minimum-threshold
    # https://www.researchgate.net/publication/323362742_Keypoint_Descriptors_in_SIFT_and_SURF_for_Face_Feature_Extractions
    surf = cv2.xfeatures2d.SURF_create(1000)

    for file in files:

        instance = cv2.imread(dirname + file)

        if instance is None:
            return False
        
        keypoints, descriptors = surf.detectAndCompute(instance, None)
        dataset.append((file, keypoints, descriptors))
        n_keypoints.append(len(keypoints))

    if verbose:
        print('average number of extracted keypoints:', np.mean(n_keypoints))

    return True


def painting_retrieve(paintings):

    paintings_data = []

    for painting, box in paintings:

        # SURF for frame
        surf = cv2.xfeatures2d.SURF_create(1000)
        keypoints_1, descriptors_1 = surf.detectAndCompute(painting, None)

        # BF matcher
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

        # we use candidate as a container to store the best matches
        candidates = []
        smallest_detected_sum_of_distances = 100000000000.0


        # loop through all the dataset images to find the most similar
        for data in dataset:

            keypoints_2 = data[1]
            descriptors_2 = data[2]

            matches = bf.knnMatch(descriptors_2, descriptors_1, k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good.append(m)

            matches = bf.match(descriptors_1, descriptors_2)
            matches = sorted(matches, key=lambda x: x.distance)

            matches = matches[:20]
            distance_sum = sum(match.distance for match in matches)


            # Only append to results if the match is likely to be more similar to speed up lookup
            if distance_sum < smallest_detected_sum_of_distances and len(good) > 5:
                smallest_detected_sum_of_distances = distance_sum
                candidates.append((distance_sum, data[0]))
            


        # if there's not any candidate we assume the painting is not in the DB
        if not candidates:
            paintings_data.append((painting, box, None))
        else:

            # took the best from all candidates
            _, best = min(enumerate(candidates), key=itemgetter(1))
            #print('el:', best[0])

            # threshold in order to discard false positive 
            if best[0] >= 20.0:
                paintings_data.append((painting, box, None))
            else:

                with open('resources/data.csv', 'r+', encoding="utf-8") as file:

                    paintings_info = csv.reader(file)
                    for info in paintings_info:
                        painting_num = info[3]
                        if painting_num == best[1]:
                            title = info[0].encode('utf-8')
                            room = info[2]
                            num = painting_num
                            best_score = best [0]
                            paintings_data.append((painting, box, (title, room, num, best_score)))

    return paintings_data