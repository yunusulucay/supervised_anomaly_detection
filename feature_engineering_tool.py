import json
import pickle
import bisect

class feature_engineering():
    def __init__(self, df):
        with open("feature_engineering_tools/configuration.json", "r") as open_file:
            self.configuration = json.load(open_file)
        
        self.get_pickles()
        
        self.df = df
        
    def full_process(self):
        self.drop_columns()
        
        self.label_encoding()
        
        self.normalizing_columns()
        return self.df
    
    def drop_columns(self):
        self.df.drop(self.configuration["drop_columns"], axis=1, inplace=True)
        return self.df
    
    def normalizing_columns(self):
        for col in self.configuration["numerical_columns"]:
            self.df.loc[:,col] = self.standard_scaler_dict[f"ss_{col}"].transform(self.df[[col]])
        return self.df
    
    def label_encoding(self):
        # Label encoding. Additional processes for if the value in train data not in test.
        
        for col in self.configuration["categorical_columns"]:
            temporary = self.df[col].map(lambda s: "other" if s not in self.label_encoder_dict[f"le_{col}"].classes_ else s)
            label_encoder_classes = self.label_encoder_dict[f"le_{col}"].classes_.tolist()
            
            bisect.insort_left(label_encoder_classes, "other")
            label_encoder_classes = list(set(label_encoder_classes))
            
            self.label_encoder_dict[f"le_{col}"].classes_ = label_encoder_classes
            
            self.df.loc[:,col] = self.label_encoder_dict[f"le_{col}"].transform(temporary)
        return self.df
    
    def get_pickles(self):
        with open(self.configuration["label_encoder_path"], "rb") as le_pickle:
            self.label_encoder_dict = pickle.load(le_pickle)

        with open(self.configuration["standart_scaler_path"], "rb") as ss_pickle:
            self.standard_scaler_dict = pickle.load(ss_pickle)
        
        return self.label_encoder_dict, self.standard_scaler_dict
    
    def print_df_columns(self):
        print(self.df.columns)