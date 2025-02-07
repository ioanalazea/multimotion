# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 16:26
# @Author  : Xiaoyi Sun
# @Site    : 
# @File    : old_exp_data_converter.py
# @Software: PyCharm
import pandas as pd
import re


def convert_old_experiment(survey_path, df_out_path):
    def prepare_survey_data():
        survey_data_raw = []
        f = open(survey_path)
        counter = 0
        index_questions, index_responses, index_responses_headers = -1, -1, -1
        questions = None

        for line in f:
            line = line.strip()
            if len(line) > 0:
                line = line.replace(' \t ', ';').replace('\t', ';')
                if line.find("Matrix contains") != -1:
                    index_questions = counter
                elif line.find("Response Matrix") != -1:
                    index_responses = counter
                elif line.find("STUDY") != -1:
                    index_responses_headers = counter
            line = line.split(';')
            survey_data_raw.append(line)
            counter += 1

        f.close()

        if index_questions != -1:
            questions = pd.DataFrame(survey_data_raw[index_questions + 3:index_responses - 1])
            no_columns_needed = len(survey_data_raw[index_questions + 3]) - len(survey_data_raw[index_questions + 2])
            questions.columns = survey_data_raw[index_questions + 2] + ["Label-info"] * no_columns_needed

        responses = pd.DataFrame(survey_data_raw[index_responses_headers + 1:])
        responses.columns = survey_data_raw[index_responses_headers]

        return questions, responses

    _, responses = prepare_survey_data()

    slides = sorted(
        set(i[i.find("_") + 1: i.rfind("_")] for i in responses.loc[:, ['LABELID' in i for i in responses.columns]]))

    elements_to_remove = ["Survey_L", "Survey_I", "Survey_Y"]
    slides = [slide for slide in slides if slide not in elements_to_remove]

    emotions = sorted(set(
        i[i.find('"') + 1: i.rfind('"')] for i in
        responses.loc[:, [next(iter(slides)) in i for i in responses.columns]]))

    # FYI: A-1 survey is actually for A4 video

    df = pd.DataFrame()
    for respondent in responses["RESPONDENT"]:
        df_stimuli = pd.DataFrame({"respondent": respondent, "stimulus": slides})
        responses_respondent = responses.loc[responses["RESPONDENT"] == respondent]
        for emotion in emotions:
            values = []
            for slide in slides:
                value = responses_respondent.loc[:,
                        [True if re.search("LABELVALUE_" + slide + "_" + "[\\w\\W]*" + emotion, column)
                         else False for column in responses.columns]].values[0][0]
                if value == "" or value is None:
                    values.append(0)
                else:
                    values.append(int(value) - 1)
            df_emotion = pd.DataFrame({emotion: values})
            df_stimuli = pd.concat([df_stimuli, df_emotion], axis=1)
        df = pd.concat([df, df_stimuli], ignore_index=True)

    name_mapping = {
        "Survey_A": "A_HN",
        "Survey_A1": "A1_LP",
        "Survey_AA2": "A2_LP",
        "Survey_AA3": "A3_LP",
        "Survey_B": "B_HN",
        "Survey_C": "C_LN",
        "Survey_A-1": "A4_LP",
        "Survey_F": "F_HN",
        "Survey_G": "G_HP",
        "Survey_H": "H_HP",
        "Survey_J": "J_Ne",
        "Survey_K": "K_Ne",
        "Survey_M": "M_LN",
        "Survey_N": "N_LN",
        "Survey_O": "O_LN",
        "Survey_P": "P_HP",
        "Survey_U": "U_Ne",
        "Survey_V": "V_Ne",
        "Survey_W": "W_HN",
        "Survey_q": "Q_HP"
    }

    df['stimulus'] = df['stimulus'].replace(name_mapping)
    df.sort_values(by=['respondent', 'stimulus'], inplace=True)
    df.to_csv(df_out_path, index=False)



