import shutil

base_dir = "/mnt/ShareGPT4Video/unzip_folder/"
new_id = 1
with open("/mnt/workspace/SVD_Xtend-main/static_dataset/static.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        image_path = base_dir+line
        print(image_path)
        new_path = "/mnt/workspace/SVD_Xtend-main/static_dataset/static/"#+str(new_id)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        new_name = new_path+'/video_'+str(new_id)+'.mp4'
        print(new_name)
        shutil.copyfile(image_path,new_name)
        new_id+=1