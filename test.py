import cv2
import cv
import numpy as np
from constants import *
from image import *
import matplotlib.pyplot as plt
import os
from heapq import *
from edge_scores import *
from DSFNode import *
import sys
import copy
import string
import constants

def dsf_reconstruct_dataset():
	type2 = True
	# random.seed(1122334455667)

	direct_metric_vals = {}
	neighbor_metric_vals = {}

	for f in os.listdir(img_folder):
		print f
		picName = f[:string.rfind(f, '.')]
		reconDir = './Images/Reconstruction_Generations/' + picName + '/'
		if not os.path.exists(reconDir):
			os.makedirs(reconDir)

		img = Image(img_folder + f)
		scramble(img, type2)
		assemble_image(img, reconDir + 'beginning.jpg')
		K = len(img.pieces)
		for sq in img.pieces:
			sq.compute_mean_and_covar_inv()

		dists_mgc = compute_edge_dist(img, type2)

		dists_list = []
		for i in xrange(K):
			for edgeNum_i in xrange(4):
				for j in xrange(K):
					for edgeNum_j in xrange(4):
						dists_list.append( (dists_mgc[i, edgeNum_i, j, edgeNum_j], i, j, edgeNum_i, edgeNum_j) )
		heapify(dists_list)

		forest = DisjointSetForest(K)

		print 'starting with', len(dists_list), 'edges'

		while forest.numClusters > 1:
			edge = heappop(dists_list)
			clust_index = forest.union(edge[1], edge[2], edge[3], edge[4])
			if clust_index != -1:
				num_c = forest.numClusters

				print 'reconstructing', num_c, edge
				pixels = forest.reconstruct(clust_index, img.pieces)
				# cv2.imwrite(reconDir + 'gen' + str(K - num_c) + '.cluster' + str(clust_index) + '.jpg', pixels)

		print 'ended with', len(dists_list), 'edges'

		for index in forest.pieceCoordMap.keys():
			pixels = forest.reconstruct(index, img.pieces)
			cv2.imwrite(reconDir + 'pre_trim.jpg', pixels)

		index_grid, occupied_grid = forest.get_orig_trim_array()
		pixels = forest.reconstruct_trim(index_grid, img.pieces)

		frame_W = img.W / P
		frame_H = img.H / P
		trim_index_grid, extra_piece_list = forest.trim(index_grid, occupied_grid, frame_W, frame_H)

		pixels = forest.reconstruct_trim(trim_index_grid, img.pieces)
		cv2.imwrite(reconDir + 'trim_index_grid.jpg', pixels)
		filled_index_grid = forest.fill(trim_index_grid, extra_piece_list, img)
		pixels = forest.reconstruct_trim(filled_index_grid, img.pieces)

		cv2.imwrite(reconDir + 'filled_index_grid.jpg', pixels)

		correct = direct_metric(img, filled_index_grid)
		direct_metric_vals[picName] = correct
		print picName, 'direct', correct, '/', frame_W*frame_H
		correct = neighbor_metric(img, filled_index_grid)
		print picName, 'neighbor', correct, '/', 2*frame_W*frame_H - frame_H - frame_W
		neighbor_metric_vals[picName] = correct

	print 'direct metric', direct_metric_vals
	print 'neighbor metric', neighbor_metric_vals

def dsf_reconstruction_test(img_filename):



	type2 = False
	random.seed(1122334455667)
	picName = img_filename[string.rfind(img_filename, '/')+1:string.rfind(img_filename, '.')]
	reconDir = './Images/Reconstruction_Generations_' + str(P) + '_t/' + picName + '/'
	print reconDir
	if not os.path.exists(reconDir):
		os.makedirs(reconDir)

	img = Image(img_filename)
	frame_W = img.W / P
	frame_H = img.H / P
	min_edges = min(constants.min_edges, 2*frame_W*frame_H - frame_H - frame_W)
	scramble(img, type2)
	assemble_image(img, reconDir + 'beginning.jpg')
	K = len(img.pieces)
	for sq in img.pieces:
		sq.compute_mean_and_covar_inv()

	# dists_mgc = compute_edge_dist(img, type2)
	# np.save('./Data/6', dists_mgc)
	# dists_mgc = np.load('./Data/6.npy')

	# dists_list = []
	orig_dists_list = compute_edge_dist(img, type2)
	# for i in xrange(K):
	# 	for edgeNum_i in xrange(4):
	# 		for j in xrange(K):
	# 			for edgeNum_j in xrange(4):
	# 				dists_list.append( (dists_mgc[i, edgeNum_i, j, edgeNum_j], i, j, edgeNum_i, edgeNum_j) )
	best_extra = frame_W * frame_H
	best_forest = None
	best_grid = None
	best_extra_list = None
	threshold = frame_W * frame_H // 8 * 3
	rep = 0
	LR_W = 1
	UD_W = 1
	while rep < 11 or best_extra > threshold:
		rep += 1
		if rep == 32:
			break
		if LR_W + UD_W > 100:
			return
		dists_list = copy.deepcopy(orig_dists_list)
		mag = random.random() * 0.1
		if rep == 1:
			mag = 0
		print 'mag', mag, ', LR, UD', LR_W, UD_W
		for dist in dists_list:
			if dist[3] % 2  == 0:
				dist[0] += random.random() * mag * LR_W + 0.01 * LR_W
			else:
				dist[0] += random.random() * mag * UD_W + 0.01 * UD_W
		heapify(dists_list)

		forest = DisjointSetForest(K)

		print 'starting with', len(dists_list), 'edges'
		cnt = [0] * 2

		while forest.numClusters > 1 and len(dists_list) > 0:
			edge = heappop(dists_list)
			clust_index = forest.union(edge[1], edge[2], edge[3], edge[3])
			if clust_index != -1:
				num_c = forest.numClusters
				cnt[edge[3] % 2] += 1

				if num_c < 50:
					print 'reconstructing', num_c, edge
				pixels = forest.reconstruct(clust_index, img.pieces)
				# cv2.imwrite(reconDir + 'gen' + str(K - num_c) + '.cluster' + str(clust_index) + '.jpg', pixels)

		print 'ended with', len(dists_list), 'edges'
		if len(dists_list) == 0:
			continue

		forest.collapse()

		index_grid, occupied_grid = forest.get_orig_trim_array()
		index_grid = index_grid.astype(int)
		print index_grid.shape
		
		pixels = forest.reconstruct_trim(index_grid, img.pieces)


		trim_index_grid, extra_piece_list = forest.trim(index_grid, occupied_grid, frame_W, frame_H)
		if trim_index_grid.shape[0] < frame_H or trim_index_grid.shape[1] < frame_H:
			rep -= 1
			print 'bad trim grid'
			print 'cnt: ', cnt[0], cnt[1]
			'''
			if rep == 0:
				if cnt[1] < cnt[0]:
					LR_W += 1
				elif cnt[0] < cnt[1]:
					UD_W += 1
			else:
				if cnt[1] * 2 < cnt[0]:
					LR_W += 1
				elif cnt[0] * 2 < cnt[1]:
					UD_W += 1
			'''
			if trim_index_grid.shape[0] < frame_H:
				UD_W += 1
			else:
				LR_W += 1
			continue
		elif cnt[1] * 2 < cnt[0]:
			LR_W += 1
		elif cnt[0] * 2 < cnt[1]:
			UD_W += 1
		print 'iteration done', len(extra_piece_list), 'pieces out of place'
		
		if len(extra_piece_list) < best_extra:
			best_extra, best_extra_list, best_forest, best_grid = len(extra_piece_list), extra_piece_list, forest, trim_index_grid
			if rep > 15:
				break

	if best_grid is None:
		return
	pixels = best_forest.reconstruct_trim(best_grid, img.pieces)
	cv2.imwrite(reconDir + 'trim_index_grid.jpg', pixels)
	filled_index_grid = forest.fill(best_grid, best_extra_list, img)
	pixels = best_forest.reconstruct_trim(filled_index_grid, img.pieces)
	cv2.imwrite(reconDir + 'filled_index_grid.jpg', pixels)

	correct = direct_metric(img, filled_index_grid)
	answer = []
	for i in xrange(K):
		answer.append(str(filled_index_grid[frame_H - 1 - i // frame_W, i % frame_W, 0]))
		
	with open('./' + picName + '_out.txt', 'w') as f:
  		f.write(' '.join(answer))

def mixed_test():
	type2 = True
	random.seed(1122334455667)
	picName = img_filename[string.rfind(img_filename, '/')+1:string.rfind(img_filename, '.')]
	picName2 = img_filename2[string.rfind(img_filename2, '/')+1:string.rfind(img_filename2, '.')]
	reconDir = './Images/Reconstruction_Generations/' + picName + 'mixedWith' +  picName2 + '/'
	if not os.path.exists(reconDir):
		os.makedirs(reconDir)

	img = Image(img_filename)
	scramble(img, type2)
	K = len(img.pieces)
	for sq in img.pieces:
		sq.compute_mean_and_covar_inv()

	img2 = Image(img_filename2)
	# scramble(img2, type2)
	K2 = len(img2.pieces)
	for sq in img2.pieces:
		sq.compute_mean_and_covar_inv()
		sq.idx += K
	img.pieces = np.concatenate((img.pieces, img2.pieces), 1)
	K += K2

	dists_mgc = compute_edge_dist(img, type2)
	# np.save('./Data/6', dists_mgc)
	# dists_mgc = np.load('./Data/6.npy')

	dists_list = []
	for i in xrange(K):
		for edgeNum_i in xrange(4):
			for j in range(i + 1, K):
				for edgeNum_j in xrange(4):
					dists_list.append( (dists_mgc[i, edgeNum_i, j, edgeNum_j], i, j, edgeNum_i, edgeNum_j) )
	heapify(dists_list)

	forest = DisjointSetForest(K)

	print 'starting with', len(dists_list), 'edges'

	while forest.numClusters > 1:
		edge = heappop(dists_list)
		clust_index = forest.union(edge[1], edge[2], edge[3], edge[4])
		if clust_index != -1:
			num_c = forest.numClusters

			print 'reconstructing', num_c, edge
			pixels = forest.reconstruct(clust_index, img.pieces)
			cv2.imwrite(reconDir + 'gen' + str(K - num_c) + '.cluster' + str(clust_index) + '.jpg', pixels)

	print 'ended with', len(dists_list), 'edges'

	forest.collapse()

	for index in forest.pieceCoordMap.keys():
		pixels = forest.reconstruct(index, img.pieces)
		cv2.imwrite(reconDir + 'pre_trim.jpg', pixels)

def dsf_test():
	numNodes = 2000
	forest = DisjointSetForest(numNodes)

	edgeHeap = []
	for i in xrange(numNodes):
		for j in xrange(numNodes):
			if i != j:
				edgeHeap.append((random.random(), i, j, 0, 0))
	heapify(edgeHeap)

	while forest.get_num_clusters() > 1:
		edge = heappop(edgeHeap)
		i = edge[1]
		j = edge[2]
		forest.union(i, j)

	rep_0 = forest.find(0)
	rep_last = forest.find(numNodes - 1)
	print rep_0.get_order(), rep_last.get_order(), rep_0.get_data(), rep_last.get_data()

def compute_edge_correct_percents():
	type2 = True
	ssd = False
	correct_pieces_list = []
	correct_rots_list = []

	for f in os.listdir(img_folder):
		print f
		img = Image(img_folder + f)
		scramble(img, type2)
		for sq in img.pieces:
			sq.compute_mean_and_covar_inv()
		dists = compute_edge_dist(img, type2, ssd, divideBySecond=False)
		correct_pieces, correct_rots = count_correct_matches(img, dists, type2)
		correct_pieces_list.append(correct_pieces)
		correct_rots_list.append(correct_rots)
		print correct_pieces_list
		print correct_rots_list

	print np.mean(correct_pieces_list)
	print np.mean(correct_rots_list)

def save_ssd_wrong_pics():
	type2 = True
	random.seed(1122334455667)
	img = Image(img_filename)
	scramble(img, type2)
	print len(img.pieces)

	for sq in img.pieces:
		sq.compute_mean_and_covar_inv()

	dists_mgc = compute_edge_dist(img, type2)
	dists_ssd = compute_edge_dist(img, type2, ssd=True)

	K = len(img.pieces)
	count = 0

	for i in xrange(K):
		square = img.pieces[i]
		for ri in xrange(4):
			square_total_rot = (square.rot_idx + ri) % 4

			# only consider the pieces which actually have a match in the picture
			if not img.square_has_neighbor(square, square_total_rot):
				continue

			min_ind_mgc, min_rot_mgc, _ = get_nearest_edge(i, ri, dists_mgc, K, type2)
			min_s_mgc = img.pieces[min_ind_mgc]
			min_s_total_rot_mgc = (min_s_mgc.rot_idx + min_rot_mgc) % 4
			cor_piece_mgc, cor_rot_mgc = is_correct_neighbor(square, square_total_rot, min_s_mgc, min_s_total_rot_mgc)
			if not (cor_piece_mgc and cor_rot_mgc):
				continue

			min_ind_ssd, min_rot_ssd, _ = get_nearest_edge(i, ri, dists_ssd, K, type2)
			min_s_ssd = img.pieces[min_ind_ssd]
			min_s_total_rot_ssd = (min_s_ssd.rot_idx + min_rot_ssd) % 4
			cor_piece_ssd, cor_rot_ssd = is_correct_neighbor(square, square_total_rot, min_s_ssd, min_s_total_rot_ssd)
			if cor_piece_ssd:
				continue

			save_squares(square, min_s_mgc, min_s_ssd, ri, min_rot_mgc, min_rot_ssd, 
				'./Images/differences/diff' + str(count) + '.jpg')
			count += 1


#img_filename = './Images/elephants/elephant.jpg'
for i in xrange(2752, 2753):
	fileName = './images/huawei_' + str(P) + '/' + str(i) + '.png'
#	fileName = '../../Pictures/data_test1_blank/' + str(P) + '/' + str(i).zfill(4) + '.png'
	dsf_reconstruction_test(fileName)

# compute_edge_correct_percents()
# dsf_reconstruction_test()
# dsf_reconstruct_dataset()
# mixed_test()
