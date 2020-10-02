"""
Created on Sat Nov 30 12:57:06 2019

@author: nimeelitha
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class Preprocessing:
    def read_csv(self,path):
        df = pd.read_csv(path)
        df_original = df.copy()
        return df_original
    
    def random_forest(self, test_df, train_df, y_train):
        rf = RandomForestRegressor(n_estimators = 1000)
        rf.fit(train_df, y_train)
        test_df = self.compare_match(test_df, train_df)
        predictions = rf.predict(test_df)
        print(predictions)
        return predictions
        
    
    def get_cat_df(self, df):
        cat_df = df[df.select_dtypes(include='object').columns.tolist()]
        return cat_df
    
    def get_cont_df(self, df):
        cont_cols = df.describe().columns
        cont_df = df[cont_cols]
        return cont_df
    
    def fill_na(self, cat_df):
        
        #Dealing with ordinal data(making them continous)
        
        #scale_mapper_LotShape
        scale_mapper_LotShape = {'Reg':4, 'IR1':3,'IR2':2,'IR3':1}
        cat_df['LotShape'] = cat_df['LotShape'].replace(scale_mapper_LotShape)
        cat_df.LotShape.unique()
        
        #scale_mapper_Alley
        scale_mapper_Alley={'Pave':2,'Grvl':1}
        cat_df['Alley'] = cat_df['Alley'].replace(scale_mapper_Alley)
        cat_df['Alley'] = cat_df['Alley'].fillna(value=0).astype(int)
        cat_df.Alley.unique()
        
        #scale_mapper_Street
        scale_mapper_Street={'Pave':2,'Grvl':1}
        cat_df.Street=cat_df.Street.replace(scale_mapper_Street)
        cat_df.Street.unique()
        
        #scale_mapper_common
        scale_mapper_common = {'Po':1, 'Fa':2, 'TA':3,'Gd':4,'Ex':5}
        col_list=['KitchenQual','ExterCond','BsmtQual','BsmtCond','HeatingQC']
        for col in col_list:
            cat_df[col] = cat_df[col].replace(scale_mapper_common)
        cat_df['BsmtCond'] = cat_df['BsmtCond'].fillna(value=0)
        cat_df['BsmtQual'] = cat_df['BsmtQual'].fillna(value=0).astype(int)
        
        #scale_mapper_Utilities
        scale_mapper_Utilities={'AllPub':2,'NoSeWa':1,}
        cat_df.Utilities=cat_df.Utilities.replace(scale_mapper_Utilities)
        cat_df.Utilities.unique()
        
        #scale_mapper_LandSlope
        scale_mapper_LandSlope={'Gtl':3,'Mod':2,'Sev':1}
        cat_df.LandSlope=cat_df.LandSlope.replace(scale_mapper_LandSlope)
        cat_df.LandSlope.unique()
        
        #scale_mapper_BsmtExposure
        scale_mapper_BsmtExposure={'Gd':4,'Av':3,'Mn':2,'No':1}
        cat_df.BsmtExposure=cat_df.BsmtExposure.replace(scale_mapper_BsmtExposure)
        cat_df['BsmtExposure'] = cat_df['BsmtExposure'].fillna(value=0).astype(int)
        cat_df.BsmtExposure.unique()
        
        ##scale_mapper_BsmtFinType
        scale_mapper_BsmtFinType = {'GLQ':6, 'ALQ':5, 'BLQ':4,'Rec':3,'LwQ':2,'Unf':1}
        col_list=['BsmtFinType1','BsmtFinType2']
        for col in col_list:
            cat_df[col] = cat_df[col].replace(scale_mapper_BsmtFinType)
        cat_df['BsmtFinType1'] = cat_df['BsmtFinType1'].fillna(value=0).astype(int)
        cat_df['BsmtFinType2'] = cat_df['BsmtFinType2'].fillna(value=0).astype(int)
            
        #scale_mapper_Functional
        scale_mapper_Functional={'Typ':8,'Min1':7,'Min2':6, 'Mod':5, 'Maj1':4,'Maj2':3,'Sev':2,'Sal':1}
        cat_df.Functional=cat_df.Functional.replace(scale_mapper_Functional)
        cat_df.Functional.unique()
        
        #scale_mapper_FireplaceQu
        scale_mapper_FireplaceQu= {'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2, 'Na':1}
        cat_df['FireplaceQu']=cat_df['FireplaceQu'].replace(scale_mapper_FireplaceQu)
        cat_df['FireplaceQu'] = cat_df['FireplaceQu'].fillna(value=1)
        cat_df.FireplaceQu.unique()
        
        #scale_mapper_GarageCond
        scale_mapper_GarageCond= {'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2, 'Na':1}
        cat_df['GarageCond']=cat_df['GarageCond'].replace(scale_mapper_GarageCond)
        cat_df['GarageCond'] = cat_df['GarageCond'].fillna(value=1)
        cat_df.GarageCond.unique()
        
        #scale_mapper_GarageQual
        scale_mapper_GarageQual= {'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2, 'Na':1}
        cat_df['GarageQual']=cat_df['GarageQual'].replace(scale_mapper_GarageQual)
        cat_df['GarageQual'] = cat_df['GarageQual'].fillna(value=1)
        cat_df.GarageQual.unique()
        
        #scale_mapper_GarageFinish
        scale_mapper_GarageFinish= { 'Fin':4, 'RFn':3, 'Unf':2, 'Na':1}
        cat_df['GarageFinish']=cat_df['GarageFinish'].replace(scale_mapper_GarageFinish)
        cat_df['GarageFinish'] = cat_df['GarageFinish'].fillna(value=1)
        cat_df.GarageFinish.unique()
        
        #scale_mapper_PavedDrive
        scale_mapper_PavedDrive= { 'Y':3, 'P':2, 'N':1}
        cat_df['PavedDrive']=cat_df['PavedDrive'].replace(scale_mapper_PavedDrive)
        cat_df.PavedDrive.unique()
        
        #scale_mapper_GarageType
        scale_mapper_GarageType= {'2Types':7, 'Attchd':6, 'Basment':5, 'BuiltIn':4, 'CarPort':3, 'Detchd':2, 'Na':1}
        cat_df['GarageType']=cat_df['GarageType'].replace(scale_mapper_GarageType)
        cat_df['GarageType'] = cat_df['GarageType'].fillna(value=1)
        cat_df.GarageType.unique()
        
        #scale_mapper_PoolQC
        scale_mapper_PoolQC= {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Na':1}
        cat_df['PoolQC']=cat_df['PoolQC'].replace(scale_mapper_PoolQC)
        cat_df['PoolQC'] = cat_df['PoolQC'].fillna(value=1)
        cat_df.PoolQC.unique()
        
        #scale_mapper_Fence
        scale_mapper_Fence= {'GdPrv':3, 'MnPrv':2, 'GdWo':3, 'MnWw':2, 'Na':1}
        cat_df['Fence']=cat_df['Fence'].replace(scale_mapper_Fence)
        cat_df['Fence'] = cat_df['Fence'].fillna(value=1)
        cat_df.Fence.unique()
        
        #Replacing NA with NM to avoid nulls
        cat_df['MiscFeature'] = cat_df['MiscFeature'].fillna(value = 'NM')
        cat_df.MiscFeature.unique()
        
        #replacing all categorical null values with mode
        cat_columns = cat_df.select_dtypes(include='object').columns.tolist()
        for col in cat_columns:
            cat_df[col].fillna(cat_df[col].mode()[0],inplace=True)
            
        print('category isnull?? : ', cat_df[cat_columns].isnull().values.any())#, cat_df[cat_columns].isnull().sum())
        
        #replacing all continuous null values with mean
        cont_columns = cat_df.describe().columns
        for col in cont_columns:
            cat_df[col] = cat_df[col].fillna(cat_df[col].mean())
        print('continuous isnull?? : ', cat_df[cont_columns].isnull().values.any())#, cat_df[cont_columns].isnull().sum())

        return cat_df
    
    #adding more derived features
    def add_features(self, df):
        df['Total_Bathrooms'] = df['BsmtFullBath'] + df['FullBath'] + 0.5*(df['BsmtHalfBath'] + df['HalfBath'])
        df['Age_of_House'] = df['YrSold']- df['YearBuilt']
        df['Age_of_remodeling'] = df['YrSold'] - df['YearRemodAdd']
        df['Garage_Age'] = df['YrSold'] - df['GarageYrBlt']
        df.drop(columns = ['YrSold', 'YearBuilt', 'YearRemodAdd','GarageYrBlt'],inplace=True)
        return df
    
    #creating and storing corr dataframe
    def create_corr_df(self, df):
        cont_cols = df.describe().columns
        cont_df = df[cont_cols]
        corr = cont_df.corr()       
        corr.to_excel('./correlation.xlsx')
        return corr
    
    #creating a dict of all pairs of features with correlation higher than threshold
    def get_corr_features(self, corr, threshold):
        highcorr_dict = {}
        for col in corr.columns:
            highcorr_dict[col] = []
            for cor, value in corr[col].iteritems():
                if abs(value) >= threshold:
                    highcorr_dict[col].append(cor)
        return highcorr_dict 
    #removing some features based on Pearson's correlation
    '''def drop_features(self, df):
        df.drop(columns = ['TotRmsAbvGrd'], inplace=True)
        df.drop(columns = ['1stFlrSF'], inplace=True)
        df.drop(columns = ['BedroomAbvGr'], inplace=True)
        df.drop(columns = ['LotFrontage'], inplace=True)
        df.drop(columns = ['OpenPorchSF'], inplace=True)
        df.drop(columns = ['WoodDeckSF'], inplace=True)
        return df'''
    # MAPE got decreased from 10.4 to 10.15 after dropping these columns

    
    #normalizing sales price
    def normalize_price(self, df):
        df['SalePrice'] = np.log1p(df['SalePrice'])
        return df
    
    #creating dummy varables and droping its categorical column
    def create_dummy_values(self, df):
        dummy_cols = df.select_dtypes(include='object').columns.tolist()
        for col in dummy_cols:
            dummies = pd.get_dummies(df[col], drop_first=True)
            columns = dummies.columns
            columns_dict = {}
            for column in columns:
                columns_dict[column] = col + '_' + column
            dummies = dummies.rename(columns = columns_dict)
            df = df.merge(dummies, left_index = True, right_index = True)
            df = df.drop(columns = [col])
            
        return df
    
    def compare_match(self, test_df, train_df):
        missing_cols = {}
        for col in train_df.columns:
            if col not in test_df.columns:
                missing_cols[col] = [0] * len(test_df)
        missing_col_df = pd.DataFrame.from_dict(missing_cols)
        test_df = test_df.merge(missing_col_df, left_index = True, right_index = True )
        return test_df
        
    
    
if __name__ == "__main__":
    
    from sklearn import linear_model
    
    preprocessing = Preprocessing()
    train_path = './train.csv'
    test_path = './test.csv'
    actual_price_path = './test_actual_price.csv'
    
    train_df = preprocessing.read_csv(train_path)
    test_df = preprocessing.read_csv(test_path)
    
    train_df = preprocessing.fill_na(train_df)
    test_df = preprocessing.fill_na(test_df)
    
    train_df = preprocessing.add_features(train_df)
    test_df = preprocessing.add_features(test_df)
    
    train_df = preprocessing.create_dummy_values(train_df)
    test_df = preprocessing.create_dummy_values(test_df)
    
    y_train = train_df['SalePrice']
    y_train = y_train.apply(pd.to_numeric, errors='coerce')
    train_df.drop(columns=['SalePrice'], inplace=True)
    
    predictions = preprocessing.random_forest(test_df, train_df, y_train)

    
#    model = linear_model.LinearRegression()
#    model.fit(train_df, y_train)
#    
#    matched_test_df = preprocessing.compare_match(test_df, train_df)
#    
#    predictions = np.absolute(model.predict(matched_test_df))
   
    actual_price = preprocessing.read_csv(actual_price_path).sort_values(by = 'Id')
    actual_price['Predicted_SalePrice'] = list(predictions)
    mape = np.mean(np.abs(actual_price['SalePrice'] - actual_price['Predicted_SalePrice']) / actual_price['SalePrice'])
    print(mape)