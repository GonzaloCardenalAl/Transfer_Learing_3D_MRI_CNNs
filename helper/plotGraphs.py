#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 12:01:35 2018

@author: roshan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from copy import deepcopy


from scipy.ndimage.interpolation import zoom
import nibabel
import nibabel as nib
import nilearn.image as nilimg


def plotGraph(df, cat, plt_type='bar', 
              dropna=False, figsize=(6,4), title=None, ax=None,
              bins=10, print_bar_count=True): 
    '''
    Plots a graph of type defined in 'plt_type' on the data in the column given by 'cat'.
    Arg::
    plt_type: can be one of 'hist+density', 'hist', 'pie', 'bar', 'barh'  
    '''   
    sns.set(color_codes=False)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    #set title
    if(title):
        ax.set_title(title)
    else:
        description = cat.split(" Index")[0]
        ax.set_title(description)
    
    data = df[cat]
    # if nans in the df then replace
    if(dropna): 
        data = data.dropna()
        data = data[(data!='NaN') or (data!="missing")]
    
    # histogram and density plots
    if (plt_type == 'hist+density'):
        data.plot.hist(grid = True, alpha=0.7, density=True, normed=True, bins=bins, ax=ax)
        data.plot.density(legend=True)
        ax.set_ylabel("density of subjects")
        
    # only histogram plots
    elif (plt_type == 'hist'):
        data.astype(float).plot.hist(grid=True, alpha=0.7, bins=bins, ax=ax)
        ax.set_ylabel("number of subjects")

         # pie-chart plots
    elif (plt_type == 'pie'):            
        val = dict(data.astype('str').value_counts())
        ax.pie(list(val.values()), labels = list(val.keys()), autopct='%1.1f%%', shadow=True, startangle=270) 
        ax.axis('equal')
        
    # bar plot
    elif (plt_type == 'bar'): 
        data.value_counts(dropna=dropna).sort_index().plot.bar(sort_columns=True, grid=True, rot=75, ax=ax,  width=0.9)
        if print_bar_count:
            x_texts = list(data.value_counts().sort_index())
            for i, x in enumerate(x_texts):
                ax.text( i , x+(x/100)+1 , str(x))
        ax.set_ylabel("number of subjects")
        
    # barh plot
    elif (plt_type == 'barh'): 
        data.value_counts(dropna=dropna).sort_index().plot.barh(ax=ax, width=0.9)
        
        if print_bar_count:
            y_texts = list(data.value_counts().sort_index())
            for i, y in enumerate(y_texts):
                ax.text(y+(y/100), i, str(y))
        ax.set_xlabel("number of subjects")
    
    else:
        raise ValueError("Invalid value for plt_type. It can be one of 'hist+density', 'hist', 'pie', 'bar', barh") 
        
        
        
######################## UKBiobank: ICD related plotting functions ################################################################

ICD10map = {
    "A00-B99":"Certain infectious and parasitic diseases",
    "C00-D48":"Neoplasms",
    "D50-D89":"Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism",
    "E00-E90":"Endocrine, nutritional and metabolic diseases",
    "F00-F99":"Mental and behavioural disorders",
    "G00-G99":"Diseases of the nervous system",
    "H00-H59":"Diseases of the eye and adnexa",
    "H60-H95":"Diseases of the ear and mastoid process",
    "I00-I99":"Diseases of the circulatory system",
    "J00-J99":"Diseases of the respiratory system",
    "K00-K93":"Diseases of the digestive system",
    "L00-L99":"Diseases of the skin and subcutaneous tissue",
    "M00-M99":"Diseases of the musculoskeletal system and connective tissue",
    "N00-N99":"Diseases of the genitourinary system",
    "O00-O99":"Pregnancy, childbirth and the puerperium",
    "P00-P96":"Certain conditions originating in the perinatal period",
    "Q00-Q99":"Congenital malformations, deformations and chromosomal abnormalities",
    "R00-R99":"Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
    "S00-T98":"Injury, poisoning and certain other consequences of external causes",
    "V01-Y98":"External causes of morbidity and mortality",
    "Z00-Z99":"Factors influencing health status and contact with health services",
    "U00-U99":"Codes for special purposes" }

ICD10_G_map = {
    "G10-G14":"Inflammatory diseases of the central nervous",
    "G20-G26":"Systemic atrophies primarily affecting the central nervous",
    "G30-G32":"Other degenerative diseases of the nervous system",
    "G35-G37":"Demyelinating diseases of the central nervous system",
    "G40-G47":"Episodic and paroxysmal disorders",
    "G50-G59":"Nerve, nerve root and plexus disorders",
    "G60-G64":"Polyneuropathies and other disorders of the peripheral nervous system",
    "G70-G73":"Diseases of myoneural junction and muscle",
    "G80-G83":"Cerebral palsy and other paralytic syndromes",
    "G90-G99":"Other disorders of the nervous system" }

ICD10_F_map = {
    "F00-F09":"Organic, including symptomatic, mental disorders",
    "F10-F19":"Mental and behavioural disorders due to psychoactive substance use",
    "F20-F29":"Schizophrenia, schizotypal and delusional disorders",
    "F30-F39":"Mood [affective] disorders",
    "F40-F48":"Neurotic, stress-related and somatoform disorders",
    "F50-F59":"Behavioural syndromes associated with physiological disturbances and physical factors",
    "G60-G64":"Polyneuropathies and other disorders of the peripheral nervous system",
    "F60-F69":"Disorders of adult personality and behaviour",
    "F70-F79":"Mental retardation",
    "F80-F89":"Disorders of psychological development",
    "F90-F98":"Behavioural and emotional disorders with onset usually occurring in childhood and adolescence",
    "F99-F99":"Unspecified mental disorder"}

def plotIcd10Distribution(df, CategoryMap=lambda x:x[:3], 
                          threshold=0, exclude_cats=[],
                          pieChart=False, title=None, rot=60):
    '''
Maps the categories using the 'CategoryMap' function to parent categories like in the 'ICD10map' mentioned above
and then plots a bar chart and a pie chart of the distribution of these mapped categories
if 'threshold' is given, Categories which have a count lower than or equal to the threshold are dropped'''
    cat_counter = Counter()
    
    assert "Diagnoses - all ICD10" in df.columns, "this func only works on if df has a column 'Diagnoses - all ICD10' that contain list of ICD diagnosis codes per subject"
    for each_subject in df["Diagnoses - all ICD10"]:
        #get a list of mapped ICD parent categories. remove duplicate parent categories
        parent_cats = list(set([CategoryMap(each_cat) for each_cat in each_subject]))
        # count each category
        for each_parent_cat in parent_cats:
            cat_counter[each_parent_cat] += 1
            
    #remove categories whose counts are lower than the threshold given
    cat_counter_copy = deepcopy(cat_counter)
    for cat, count in cat_counter_copy.items():
        catname = 'none' if cat is None else cat.lower()
        if(count <= threshold) or np.any([exclude in catname for exclude in exclude_cats]):
            del cat_counter[cat]
    
    #ignore 'None' category if it exists    
    try:
        del cat_counter[None]
    except KeyError:
        pass
    
    # convert the counts to a pandas dataframe for easy plotting
    ICD10_count_df = pd.DataFrame.from_dict(cat_counter, 'index').sort_index()
    ICD10_count_df.columns = ["number of subjects"]
    
    #bar plot
    f, ax = plt.subplots(figsize=(10,8))
    ICD10_count_df.plot.bar(grid=True, rot=rot, ax=ax)
    txt_pos = max(ICD10_count_df["number of subjects"]) //100
    for i, x in enumerate(ICD10_count_df["number of subjects"]):
        ax.text( i , x + txt_pos, x)
    
    if(title):
        ax.set_title(title)
    
    #pie plot
    if(pieChart):
        f2, ax2 = plt.subplots(figsize=(10,8))
        ax2.pie(list(ICD10_count_df["number of subjects"]), labels = ICD10_count_df.index,
                autopct='%1.1f%%', shadow=True, startangle=90) 
        ax2.axis('equal')
    
    del(ICD10_count_df)
    
    return cat_counter


'''Takes a string denoting a ICD10 category ex. "S91" and returns a numeric value for comparison'''
def getICDVal(strVal):
    letter = ord(strVal[0])*100
    num = int(strVal[1:])
    return (letter + num)


'''Takes a string range of the ICD10 categories ex. "V01-Y98" and returns the range in numeric form for comparison'''
def getICDRange(strRange):
    l,u = strRange.split("-")
    return range(getICDVal(l), getICDVal(u)+1)

'''Given the numeric version of a ICD10 category, this func returns 
the parent category and the description of the category'''
def mapICDParents(category):
    # first check if the category is empty
    if(category is np.nan):
        return "NaN", "No data / Healthy"
    category = category[:3]
    for key in ICD10map.keys():
        if getICDVal(category) in getICDRange(key):
            return key
    # raise error if an invalid category is given
    raise ValueError("Invalid category - {}".format(category))
    
    
# Plot distribution of 'F00-F99' category 'Mental and behavioural disorders'
def mapFCategory(category):    
    category=  category[:3] 
    for key in ICD10_F_map.keys():
        if getICDVal(category) in getICDRange(key):
            return key
    return None


def show_brain(
        img,
        cut_coords=None,
        figsize=(10, 5),
        cmap="nipy_spectral",
        draw_cross=True,
        return_fig=False,
):
    """Displays 2D cross-sections of a 3D image along all 3 axis.

    Parameters
    ----------
    img : numpy.ndarray/nibabel.NiftiImage/str
         either a 3-dimensional numpy.ndarray, a nibabel.Nifti1Image object or a path to the image file
         stored in nifTI format.
    cut_coords
        The voxel coordinates of the axes where the cross-section cuts will be performed.
        Should be a 3-tuple: (x, y, z). Default is the center = img_shape/2
    figsize
        matplotlib figsize. Default: (10,5).
    cmap
        matplotlib colormap to be used. Default: "nipy_spectral"
    draw_cross
        Draws horizontal and vertical lines which show where the cross-sections have been performed. Default: True
    return_fig
        Additionally retunrs the figure when set. Default: False

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    """

    if isinstance(img, str) and os.path.isfile(img):
        img_arr = load_nifti(img)
    elif isinstance(img, nibabel.Nifti1Image):
        img_arr = img.get_data()

    elif isinstance(img, np.ndarray):
        assert (
                img.ndim == 3
        ), "The numpy.ndarray must be 3-dimensional with shape (H x W x Z)"
        img_arr = img
    else:
        raise TypeError(
            "Invalid type provided for 'img'- {}. "
            "Either provide a 3-dimensional numpy.ndarray of a MRI image or path to "
            "the image file stored as a nifTI format.".format(
                type(img)
            )
        )

    # print(img_arr.shape)
    # img_arr = np.moveaxis(img_arr, 0, 1)
    # print(img_arr.shape)

    x_len, y_len, z_len = img_arr.shape
    # if cut_coordinates is not specified set it to the center of the image
    if cut_coords == None:
        cut_coords = (x_len // 2, y_len // 2, z_len // 2)

    f, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    ax[0].set_title("Saggital cross-section at x={}".format(cut_coords[0]))
    ax[0].imshow(
        np.rot90(img_arr[cut_coords[0], :, :]), cmap=cmap, aspect="equal"
    )
    # draw cross
    if draw_cross:
        ax[0].axvline(x=cut_coords[1], color="k", linewidth=1)
        ax[0].axhline(y=cut_coords[2], color="k", linewidth=1)

    ax[1].set_title("Coronal cross-section at y={}".format(cut_coords[1]))
    ax[1].imshow(
        np.rot90(img_arr[:, cut_coords[1], :]), cmap=cmap, aspect="equal"
    )
    ax[1].text(
        0.05,
        0.95,
        "L",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax[1].transAxes,
        bbox=dict(facecolor="white"),
    )
    ax[1].text(
        0.95,
        0.95,
        "R",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax[1].transAxes,
        bbox=dict(facecolor="white"),
    )
    # draw cross
    if draw_cross:
        ax[1].axvline(x=cut_coords[0], color="k", linewidth=1)
        ax[1].axhline(y=cut_coords[2], color="k", linewidth=1)

    ax[2].set_title("Axial cross-section at z={}".format(cut_coords[2]))
    ax[2].imshow(
        np.rot90(img_arr[:, :, cut_coords[2]]), cmap=cmap, aspect="equal"
    )
    ax[2].text(
        0.05,
        0.95,
        "L",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax[2].transAxes,
        bbox=dict(facecolor="white"),
    )
    ax[2].text(
        0.95,
        0.95,
        "R",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax[2].transAxes,
        bbox=dict(facecolor="white"),
    )
    # draw cross
    if draw_cross:
        ax[2].axvline(x=cut_coords[0], color="k", linewidth=1)
        ax[2].axhline(y=cut_coords[1], color="k", linewidth=1)

    plt.tight_layout()
    if return_fig:
        return f
