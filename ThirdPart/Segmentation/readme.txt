功能:DR病变分割(0-4级)及相应的参数计算

1.安装依赖项
pip install -r requirements.txt

2.病变分割(包含U-Net和CAMWNet这两种分割模型,分割目标包括出血,微动脉瘤,硬性渗出,软性渗出这四种病灶和视盘,加上背景共6类):
python DR_seg.py

3.参数计算(包括每类病灶每个目标中心点横纵坐标,面积,病灶个数和平均面积):
python DR_anal.py

得到U-Net_results和CAMWNet_results这两个文件夹,其中子文件夹分别为:
hemorrhages - 出血分割结果
microaneurysms - 微动脉瘤分割结果
hard_exudates - 硬性渗出分割结果
soft_exudates - 软性渗出分割结果
disc - 视盘分割结果
background - 背景分割结果
all - 所有目标分割结果
visualization - 所有目标轮廓线叠加原图结果
statistics - 包含每种病灶参数计算结果,其中test_image.csv中的3列分别为当前类病灶每个目标中心点横纵坐标及面积,summary.csv中的3列分别为图像名称,当前类病灶个数和平均面积.
