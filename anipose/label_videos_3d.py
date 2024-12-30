#!/usr/bin/env python3

from mayavi import mlab
mlab.options.offscreen = True

import numpy as np
from glob import glob
import pandas as pd
import os.path
import cv2
import sys
import skvideo.io
from tqdm import tqdm, trange
import sys
from collections import defaultdict
from matplotlib.pyplot import get_cmap
# from labutils.plotting import (sns_setup, landmark_cmap)
from .label_videos import color_scheme, cluster_name

from .common import make_process_fun, get_nframes, get_video_name, get_video_params, get_data_length, natural_keys

def connect(points, bps, bp_dict, color):
    ixs = [bp_dict[bp] for bp in bps if bp in bp_dict]
    if len(ixs): #Omit the ref point
        return mlab.plot3d(points[ixs, 0], points[ixs, 1], points[ixs, 2],
                        np.ones(len(ixs)), reset_zoom=False,
                        color=color, tube_radius=None, line_width=8) #Changed from 10 to 8
    else:
        return mlab.plot3d(0, 0, 0,
                1, reset_zoom=False,
                color=color, tube_radius=None, line_width=8) 

def connect_all(points, scheme, bp_dict, bp_to_color):
    lines = []
    for bps in scheme:
        col = tuple(float(c / 255) for c in bp_to_color[cluster_name(bps)])
        line = connect(points, bps, bp_dict, col[:-1])
        lines.append(line)
    return lines

def update_line(line, points, bps, bp_dict, bp_ignore):
    ixs = [bp_dict[bp] for bp in bps 
             if bp not in bp_ignore]
        #    if bp not in ["ref(head-post)", 
        #                  "ear(top)(right)", "ear(bottom)(right)", "ear(tip)(right)", "ear(base)(right)",
        #                   "ear(top)(left)", "ear(bottom)(left)", "ear(tip)(left)", "ear(base)(left)", 
        #                   "nose(bottom)", "nose(tip)", "nose(top)", 
        #                   "pad(top)(left)", "pad(side)(left)", "pad(center)", "pad(top)(right)", "pad(side)(right)", 
        #                   "lowerlip", "upperlip(left)", "upperlip(right)", 
        #                   "eye(front)(right)", "eye(top)(right)", "eye(back)(right)", "eye(bottom)(right)"]]
    # ixs = [bodyparts.index(bp) for bp in bps]
    new = np.vstack([points[ixs, 0], points[ixs, 1], points[ixs, 2]]).T
    line.mlab_source.points = new

def update_all_lines(lines, points, scheme, bp_dict, bp_ignore):
    for line, bps in zip(lines, scheme):
        update_line(line, points, bps, bp_dict, bp_ignore)

def retrieve_colors(cmap, face_key):
    color_group = []
    for k in cmap[0][face_key]:
        color_group.append(cmap[1][k])

    return np.array(color_group)



def visualize_labels(config, labels_fname, outname, fps=300):
    try:
        scheme = config['labeling']['scheme']
    except KeyError:
        scheme = []

    try:
        bp_ignore = config['labeling']['ignore']
    except KeyError:
        bp_ignore = []

    data = pd.read_csv(labels_fname)
    cols = [x for x in data.columns if '_error' in x]

    if len(scheme) == 0:
        bodyparts = [c.replace('_error', '') for c in cols]
    else:
        bodyparts = sorted(set([x for dx in scheme for x in dx if x not in bp_ignore]))


    bp_dict = dict(zip(bodyparts, range(len(bodyparts))))

    all_points = np.array([np.array(data.loc[:, (bp+'_x', bp+'_y', bp+'_z')])
                           for bp in bodyparts], dtype='float64')

    all_errors = np.array([np.array(data.loc[:, bp+'_error'])
                           for bp in bodyparts], dtype='float64')

    all_scores = np.array([np.array(data.loc[:, bp+'_score'])
                           for bp in bodyparts], dtype='float64')


    if config['triangulation']['optim']:
        all_errors[np.isnan(all_errors)] = 0
    else:
        all_errors[np.isnan(all_errors)] = 10000
    good = (all_errors < 100)
    all_points[~good] = np.nan

    all_points_flat = all_points.reshape(-1, 3)
    check = ~np.isnan(all_points_flat[:, 0])

    if np.sum(check) < 10:
        print('too few points to plot, skipping...')
        return
    
    low, high = np.percentile(all_points_flat[check], [5, 95], axis=0)

    nparts = len(bodyparts)
    framedict = dict(zip(data['fnum'], data.index))

    writer = skvideo.io.FFmpegWriter(outname, inputdict={
        # '-hwaccel': 'auto',
        '-framerate': str(fps), #'50', #video frame rate 
    }, outputdict={
        # '-vcodec': 'h264', 
        # TODO: Automatically choose between h264 and h264_videotoolbox if available on mac
        '-vcodec': 'libx264', 
        '-qp': '28', 
        '-pix_fmt': 'yuv420p'
    })

    cmap = config['labeling'].get('colormap', 'colorblind')
    spread = config['labeling'].get('color_spread', 0.02)
    bp_to_color = color_scheme(scheme, cmap=cmap, spread=spread)

    # cmap = get_cmap('tab10')
    # landmarks = {
    # "nose": ["nose(bottom)", "nose(tip)", "nose(top)"],
    # "whiskers(left)": ["pad(top)(left)", "pad(side)(left)", "pad(center)"],
    # "whiskers(right)": ["pad(top)(right)", "pad(side)(right)"],
    # "mouth": ["lowerlip", "upperlip(left)", "upperlip(right)"],
    # "eye(left)": ["eye(front)(left)", "eye(top)(left)", "eye(back)(left)", "eye(bottom)(left)"],
    # "eye(right)": ["eye(front)(right)", "eye(top)(right)", "eye(back)(right)", "eye(bottom)(right)"],
    # "ear(left)": ["ear(base)(left)", "ear(top)(left)", "ear(tip)(left)", "ear(bottom)(left)"],
    # "ear(right)": ["ear(base)(right)", "ear(top)(right)", "ear(tip)(right)", "ear(bottom)(right)"],
    # "ref": ["ref(head-post)"]
    # }
    # # LANDMARK_CMAP = landmark_cmap(landmarks)
    # # LANDMARK_CMAP_FLAT = get_flat_cmap(landmarks, LANDMARK_CMAP)
    # LANDMARK_CMAP_FLAT = landmark_cmap(flat=True)
    # LANDMARK_CMAP_SELECT = {k: retrieve_colors(LANDMARK_CMAP_FLAT,k) 
    #                         for k in LANDMARK_CMAP_FLAT[0].keys()}
    cmap_vis = np.zeros([len(bodyparts), 4])
    for idx, k in enumerate(bodyparts):
        cmap_vis[idx, :] = bp_to_color[k]

    points = np.copy(all_points[:, 20])
    points[0] = low
    points[1] = high

    s = np.arange(points.shape[0])
    good = ~np.isnan(points[:, 0])

    fig = mlab.figure(bgcolor=(0,0,0), size=(800,800))
    fig.scene.anti_aliasing_frames = 2

    low, high = np.percentile(points[good, 0], [10,90])
    scale_factor = (high - low) / 20.0 #Original was divided by 12.0

    mlab.clf()
    pts = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], s,
                        #color=cmap,
                        scale_mode='none', scale_factor=scale_factor)
    pts.module_manager.scalar_lut_manager.lut.table = cmap_vis
    mlab.draw()
    lines = connect_all(points, scheme, bp_dict, bp_to_color) #cmap
    print(bp_dict)
    mlab.orientation_axes()
    view = list(mlab.view())
    mlab.view(focalpoint='auto', distance='auto')
    rot_speed = np.pi / config['labeling'].get('3d_orbit_speed', 180.0)
    rot_size = config['labeling'].get('3d_orbit_size', 0.25)
    view[0] += config['labeling'].get('3d_init_azimuth', 0)
    view[1] += config['labeling'].get('3d_init_elevation', 0)
    roll = config['labeling'].get('3d_init_roll', 0)
    mlab.view(*view, reset_roll=False, roll=roll)
    orig_view = view.copy()

    for frm in trange(data.shape[0], ncols=70):
        offset = 0 #For original chewing videos of B6: 22800
        framenum = frm + offset  #num of frames to shift
        # Andrew: short videos for testing
        stop = 25800000 #final-short:23800# #1000 #25000
        if framenum > stop:
            print(f"Stopping at frame {stop}")
            exit()


        fig.scene.disable_render = True

        if framenum in framedict:
            points = all_points[:, framenum]
        else:
            points = np.ones((nparts, 3))*np.nan

        s = np.arange(points.shape[0])
        good = ~np.isnan(points[:, 0])

        new = np.vstack([points[:, 0], points[:, 1], points[:, 2]]).T
        pts.mlab_source.points = new
        update_all_lines(lines, points, scheme, bp_dict, bp_ignore)

        fig.scene.disable_render = False

        img = mlab.screenshot() 
        
        # Save a particular frame
        # if framenum == 58:
        #     filename = mlab.gcf(engine=None)
        #     mlab.savefig('test.eps', size=None, 
        #                         figure=filename, magnification='auto')


        if config['labeling'].get('3d_orbit', False):
            view[0] = orig_view[0] + np.sin(rot_speed*frm)*rot_size
            view[1] = orig_view[1] + np.cos(rot_speed*frm)*rot_size

        # Hack to understand camera orientation 
        if True:
            # For position testing
            # az = frm * 0.1 #for B6 chewing videos
            p = 3000 # slower = higher number
            # az = (30 * 2) * np.abs(frm/p - np.floor(frm/p + 0.5)) #Triangle wave
            # az = 0 # For 20240903_R27_recording_rig1 videos
            az = 90
            # roll = -85 + az # For 20240903_R27_recording_rig1 videos
            roll = az
            # roll = framenum * 0.1
            view[0] = orig_view[0] - 5 + az #add 80* before orig_view for final verison v4 view #-orig_view[0] +10 + az #
            view[1] = -(orig_view[1] + 50)
            # mlab.title(f"{framgit addenum=}, {az=}, {roll=}", color=(0, 0, 0))
            # mlab.title(f"{view[0]=}, {view[1]=}, {az=}, {roll=} ", color=(1, 1, 1))

        mlab.view(*view, reset_roll=False, roll=roll)
        writer.writeFrame(img)

    mlab.close(all=True)
    writer.close()



def process_session(config, session_path, filtered=False):
    pipeline_videos_raw = config['pipeline']['videos_raw']

    if filtered:
        pipeline_videos_labeled_3d = config['pipeline']['videos_labeled_3d_filter']
        pipeline_3d = config['pipeline']['pose_3d_filter']
    else:
        pipeline_videos_labeled_3d = config['pipeline']['videos_labeled_3d']
        pipeline_3d = config['pipeline']['pose_3d']

    video_ext = config['video_extension']

    vid_fnames = glob(os.path.join(session_path,
                                   pipeline_videos_raw, "*."+video_ext))
    orig_fnames = defaultdict(list)
    for vid in vid_fnames:
        vidname = get_video_name(config, vid)
        orig_fnames[vidname].append(vid)

    labels_fnames = glob(os.path.join(session_path,
                                      pipeline_3d, '*.csv'))
    labels_fnames = sorted(labels_fnames, key=natural_keys)

    outdir = os.path.join(session_path, pipeline_videos_labeled_3d)

    if len(labels_fnames) > 0:
        os.makedirs(outdir, exist_ok=True)

    for fname in labels_fnames:
        basename = os.path.basename(fname)
        basename = os.path.splitext(basename)[0]

        out_fname = os.path.join(outdir, basename+'.mp4')

        if os.path.exists(out_fname) and \
           abs(get_nframes(out_fname) - get_data_length(fname)) < 100:
            continue
        print(out_fname)

        some_vid = orig_fnames[basename][0]
        params = get_video_params(some_vid)
        visualize_labels(config, fname, out_fname, fps = 100) #params['fps']) <- This params... does not work anymore, idk why


label_videos_3d_all = make_process_fun(process_session, filtered=False)
label_videos_3d_filtered_all = make_process_fun(process_session, filtered=True)
