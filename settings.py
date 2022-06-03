
home = '/condo/swatwork/mcmontalbano/MYRORSS/scripts'
DATA_HOME = '/condo/swatcommon/common/myrorss'
TRAINING_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data'
multi_fields = ['MergedReflectivityQCComposite_Max_30min','MergedLLShear_Max_30min','MergedMLShear_Max_30min','Reflectivity_0C_Max_30min','MESH_Max_30min']
NSE_fields = ['MeanShear_0-6km', 'MUCAPE', 'ShearVectorMag_0-1km', 'ShearVectorMag_0-3km', 'ShearVectorMag_0-6km', 'SRFlow_0-2kmAGL', 'SRFlow_4-6kmAGL', 'SRHelicity0-1km', 'SRHelicity0-2km', 'SRHelicity0-3km', 'UWindMean0-6km', 'VWindMean0-6km', 'Heightof0C','Heightof-20C','Heightof-40C']
degrees = ['06.50', '02.50', '05.50', '01.50', '08.00', '19.00', '00.25', '00.50', '09.00', '18.00', '01.25', '20.00', '04.50', '03.50', '02.25', '07.50', '07.00', '16.00', '02.75', '12.00', '03.00', '04.00', '15.00', '11.00', '01.75', '10.00', '00.75', '08.50', '01.00', '05.00', '14.00', '13.00', '02.00', '06.00', '17.00']
targets = ['target_MESH_Max_30min']
fields_accum = [
    'MergedLLShear_Max_30min',
    'MergedMLShear_Max_30min',
    'MESH_Max_30min',
    'Reflectivity_0C_Max_30min',
    'MergedReflectivityQCComposite_Max_30min']
final_desired_fields = [
    'MergedReflectivityQC',
    'MergedLLShear_Max_30min',
    'MergedMLShear_Max_30min',
    'MESH_Max_30min',
    'Reflectivity_0C_Max_30min',
    'MergedReflectivityQCComposite_Max_30min'
    'target_MESH_Max_30min']

lon_NW, lat_NW, lon_SE, lat_SE = -130.005, 55.005, - \
    59.995, 19.995  # see MYRORSS readme at https://osf.io/kbyf7/

inmost_path = '/condo/swatcommon/common/myrorss'
UNCROPPED_HOME = '/condo/swatwork/mcmontalbano/MYRORSS/data/uncropped'
data_path = '/condo/swatwork/mcmontalbano/MYRORSS/data'
nse_path = '/condo/swatcommon/NSE'
scripts = '/condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning'
mht_path = f'{data_path}/2011/20110409'

# Hyperparameters
index = ['index', 'dataset_ID', 'loss', 'Add', 'Concat', 'dropout', 'L2', 'factor']
metrics = ['MSE','G_beta_RMSE','G_beta','POD','FAR','Delta','Hausdorff','PHD_k']
one_hot_preprocessing = ['Translation', 'Contrast', 'Rotation','Zoom','Crop','Flip','Noise']
one_hot_loss = ['MSE']


def path(date):
    return f'{data_path}/{date[:4]}/{date}'
