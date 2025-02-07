# Multimotion_groundtruth
<li>
    <h5>combine_data.py</h5>
    Merge new data and old data to updated data
</li>

<li>
    <h5>distance_matrix.py</h5>
    Compute a distance matrix ready to use for SPSS.
</li>

<li>
    <h5>final_exp_data_converter.py</h5>
    Convert the version5, final, eeg version of experiments survey data exported from imotion from txt to csv file.
</li>

<li>
    <h5>individual_map.py</h5>
    Compute individual ground truth.
</li>

<li>
    <h5>main.py</h5>
    Run this file to execute all files. You will have a list of selections. Additionally, you need to put raw group space and subject weights data in here copy and paste from word file SPSS generated.
</li>

<li>
    <h5>old_exp_data_converter.py</h5>
    Convert the versionb 1 of experiment survey data exported from imotion from txt to csv file.
</li>
<li>
    <h4>subject_weights_converter.py</h4>
    Convert raw spas group_space and subject weights data to a csv file.
</li>
<li>
    <h5>repository.txt</h5>
    List of repositories. Change only the URL accordingly.
<ol>
  <li>
  raw_data_old, URL for version 1 data only
  </li>
  <li>
  raw_data_final, URL for version 5, version final, version EEG
  </li>
  <li>
  processed_data, output csv URL and URL ready to use for distance matrix file
  </li>
  <li>
  new_data, orignial data that needs to be added to the updated dataframe
  </li>
  <li>
  new_data_2, new data that needs to be added to the updated dataframe
    </li>
  <li>
  updated_data, combined data from the 2 data above
    </li>
  <li>
  distance_matrix, output URL for the distance matrix, contains all individuals distance matrix, go to combined sheet which contains all individuals and can be used for SPSS
    </li>
  <li>
  group_space_data, URL for  group space csv file
    </li>
  <li>  
  subject_weights_data, URL for subject_weights csv file
    </li>
  <li>
  individual_ground_truth, URL for individual_ground_truth
    </li>
<ol>
</li>




<h3>How to run</h4>
<ol>
<li>
   change the repositories first, run the code in main
</li>

<li>
  you will have a list of options, select accordingly
</li>

<li>
  copy and replace the raw group space and subject weights data from SPSS of the main file where there will be 2 strings
</li>

<li>
  after computing the distance matrix, make sure you open it in excel and go to combined sheet, then save it, this will solve the empty sheet problem in SPSS
</li>

<li>
  concat the dataframes in a strict order, version 5, version 1, version final, version eeg
</li>

<ol>
