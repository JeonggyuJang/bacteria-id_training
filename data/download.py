import boto3
BUCKET_NAME = 'ramcell-largedata'
OBJECT_NAME = 'data/bacteria_ID/'
s3 = boto3.client('s3')

file_name=['wavenumbers.npy','X_2018clinical.npy','X_2019clinical.npy','X_finetune.npy','X_reference.npy','X_test.npy','y_2018clinical.npy','y_2019clinical.npy','y_finetune.npy','y_reference.npy','y_test.npy']
for i in range(11):
    print(OBJECT_NAME+file_name[i])
    s3.download_file(BUCKET_NAME, OBJECT_NAME+file_name[i],file_name[i])
