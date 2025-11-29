功能:DR分级(0-4级)及相应的Grad-CAM可视化

1.安装依赖项
pip install -r requirements.txt

2.为避免画饼状图中文字体出现乱码,需配置matplotlib包对应的字体库,参考:https://zhuanlan.zhihu.com/p/449589031?utm_id=0

3.该目录下运行示意:
rm -rf ~/.cache/matplotlib/
python demo.py --img_path test_image.jpeg --csv_path test_result.csv --dst_dir test_result
得到如下结果:
1)test_result.csv,其中7列分别为图片路径,0-4级预测概率,预测级别；
2)文件夹test_result,其中子文件夹cam,cam++,cam_gb,gb,heatmap,heatmap++,pie分别保存了Grad-CAM和Grad-CAM++叠加原图结果,引导Grad-CAM,引导反向传播,Grad-CAM和Grad-CAM++热图以及预测概率分布饼状图.
