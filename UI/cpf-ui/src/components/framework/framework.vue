<template>
  <div style="height: 100%">
   <!--
      通用的气泡提示框
    -->
    <toast-helper></toast-helper>

    <!--
      标准平板布局, 主要分为以下内容:
        1. 将左侧区域收入扩展栏
        2. 其他和PC功能一致
    -->
    <div class="phonecho" v-if="padflag">
      <!-- <van-icon name="wap-nav" @click="showPopup" size="30"/> -->
      <Button :label="$t('messages.operate')" icon="pi pi-align-justify" @click="showPopup"  v-if="padflag" class="but-1"/>
      <van-button plain type="primary" @click="changeLang1('en')" size="small" class="changelang">{{ $t('messages.en') }}</van-button>
      <van-button plain type="primary" @click="changeLang1('zh')" size="small" class="changelang">{{ $t('messages.zh') }}</van-button>
      <van-popup v-model:show="show" :style="{ width: '60%', height: '100%' }" position="left">
        <TabMenu :model="menus1" :activeItem="activeTab" @update:activeItem="onTabChange" v-if="locale == 'zh'"/>
        <TabMenu :model="menus2" :activeItem="activeTab" @update:activeItem="onTabChange" v-else/>
        <!-- 使用 <keep-alive> 缓存组件状态 -->
        <keep-alive>
          <component :is="activeTabComponent" :key="activeTab" />
        </keep-alive>
      </van-popup>
      <span class="phonecho1">{{ $t('messages.CPFAnalysisSystem') }}</span>
    </div>
   
    <!--
      标准手机布局, 主要内容如下:
        1. 其他内容和平板布局一致
        2. 在数据展示区域，增加切换区域
    -->
    <div class="phonecho" v-if="phoneflag">
      <van-icon name="wap-nav" @click="showPopup" size="30"/>
      <van-button plain type="primary" @click="changeLang1('en')" size="small" class="changelang">{{ $t('messages.en') }}</van-button>
      <van-button plain type="primary" @click="changeLang1('zh')" size="small" class="changelang">{{ $t('messages.zh') }}</van-button>
      <van-popup v-model:show="show" :style="{ width: '60%', height: '100%' }" position="left" z-index="2">
        <TabMenu :model="menus1" :activeItem="activeTab" @update:activeItem="onTabChange" v-if="locale == 'zh'"/>
        <TabMenu :model="menus2" :activeItem="activeTab" @update:activeItem="onTabChange" v-else/>
        <!-- 使用 <keep-alive> 缓存组件状态 -->
        <keep-alive>
          <component :is="activeTabComponent" :key="activeTab" />
        </keep-alive>
      </van-popup>
      <span class="phonecho1">{{ $t('messages.CPFAnalysisSystem') }}</span>
    </div>

    <!--
       标准PC布局, 主要分为以下内容:
        1. 左上的操作区域，包含:
          - 文件上传
          - 参数选择
          - 远端执行
          - 文件下载
       2. 左下上传目标选择区域
          - 一个文件列表
       3. 右上一个进度表标识区域
       4. 右下则是结果解算区域
    -->
    <Splitter class="splitter-class" ref="desktop-splitter-left">
      <!--  页面左边部分     -->
      <SplitterPanel class="flex align-items-center justify-content-center" :size="calcDesktopLeftPercent(true)" :minSize="10" v-if="!padflag && !phoneflag">
        <!--  横向320px      -->
        <Splitter layout="vertical">
          <!--  左上主事件操作区域   -->
          <SplitterPanel class="flex align-items-center justify-content-center" :size="60">
              <div style="width: 100%; height: 100%; overflow: auto">
               <!--    title area      -->
                <div style=
                         "background-image: linear-gradient(to right, var(--bluegray-500), var(--bluegray-800));
                          color: #FFFFFF; text-align: center;
                          height: 60px; line-height: 60px;
                          font-size: 1.6rem">
                    {{ $t('messages.CPFAnalysisSystem') }}
                </div>

                <div>
                  <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px; margin-right: 10px">
                    <div style="flex: 2">
                      <Badge :value="$t('messages.langChoice')" severity="info"></Badge>
                    </div>
                    <div style="flex: 4">
                      <Button :label="$t('messages.en')" link @click="changeLang1('en')"/>
                      <Button :label="$t('messages.zh')" link @click="changeLang1('zh')"/>
                    </div>
                  </div>
                </div>

                <!--   upload area   -->
                <div>
                  <upload></upload>
                </div>

                <!--   running area       -->
                <div>
                    <runner></runner>
                </div>

                <!--   download area      -->
                <div>
                  <download></download>
                </div>

              </div>
          </SplitterPanel>
          <!--   左下图像选择区       -->
          <SplitterPanel :size="40">
            <Splitter>
              <SplitterPanel class="flex align-items-center justify-content-center">
                <div style="height: 100%">
                  <user-select></user-select>
                </div>
              </SplitterPanel>
            </Splitter>
          </SplitterPanel>
        </Splitter>
      </SplitterPanel>


      <!--  右侧展示区域    -->
      <SplitterPanel :size="calcDesktopLeftPercent(false)">
        <Splitter layout="vertical" ref="desktop-splitter-right">
          <!--   右上操作提示区域       -->
          <SplitterPanel class="flex align-items-center justify-content-center" :size="calcDesktopRightPercent(true)">
              <Steps :model="items1" v-model:activeStep="activeStep" v-if="locale == 'zh'" style="font-size: small;"></Steps>
              <Steps :model="items2" v-model:activeStep="activeStep" style="font-size: small;" v-else></Steps>
          </SplitterPanel>
          <!--   右下比对区域       -->
          <SplitterPanel :size="calcDesktopRightPercent(false)">
            <Splitter>
              <SplitterPanel class="flex align-items-center justify-content-center">
                <div style="height: 100%; width: 100%; overflow-x: auto;">
                  <table-area></table-area>
                </div>
              </SplitterPanel>
            </Splitter>
          </SplitterPanel>
        </Splitter>
      </SplitterPanel>
    </Splitter>



    <van-floating-bubble icon="exchange" @click="onClick" v-if="phoneflag"/>
  </div>
</template>

<script>
import Badge from 'primevue/badge'
import upload from "@/components/upload/upload";
import Splitter from 'primevue/splitter';
import SplitterPanel from 'primevue/splitterpanel';
import Card from 'primevue/card';
import Steps from 'primevue/steps';
import runner from "@/components/runner/runner";
import {mapState} from "vuex";
import ToastHelper from "@/components/toast-helper/toast-helper";
import TableArea from "@/components/table-area/table-area";
import Download from "@/components/download/download";
import UserSelect from "@/components/user-select/user-select";
import store from "@/store";
import Button from 'primevue/button';
import Dialog from 'primevue/dialog';
import { ref } from 'vue';
import { showDialog, showToast } from 'vant';
import { useI18n } from 'vue-i18n';
import TabMenu from 'primevue/tabmenu';
import Page1 from '../tap/Page1.vue';
import Page2 from '../tap/Page2.vue';
import Page3 from '../tap/Page3.vue';
export default {
  name: "framework",
  components: {
    UserSelect,
    Download,
    TableArea,
    ToastHelper,
    Splitter,
    SplitterPanel,
    upload,
    Card,
    Badge,
    Steps,
    runner,
    Button,
    Dialog,
    TabMenu
  },
  computed: {
    ...mapState("app", ['runningStep','padflag','phoneflag', 'winSize']),
    // 根据当前选项卡返回对应的组件
    activeTabComponent() {
      switch (this.activeTab) {
        case 'page1':
          return Page1;
        case 'page2':
          return Page2;
        case 'page3':
          return Page3;
        default:
          return null;
      }
    }
  },
  watch: {
    runningStep(newVal, oldVal){
      this.activeStep = newVal
    },
    winSize(newVal, oldVal) {
      this.resizeDesktopUI("desktop-splitter-left")
      this.resizeDesktopUI("desktop-splitter-right")
    },
  },
  mounted() {
  },
  created() {

  },
  data(){
    return {
      activeStep: 0,
      items1: [
        {
          label: "上传文件"
        },
        {
          label: "点击开始按钮"
        },
        {
          label: "正在等待"
        },
        {
          label: "下载压缩包"
        }
      ],
      items2: [
        {
          label: 'Upload First'
        },
        {
          label: 'Click Start Button'
        },
        {
          label: 'Wait'
        },
        {
          label: 'Download zip'
        }
      ],
      visible: false,
      activeTab: 'page1', // 初始激活的选项卡
      menus1: [
        { label: '提交', icon: 'pi pi-upload', command: () => this.changeTab('page1') },
        { label: '选择', icon: 'pi pi-caret-right', command: () => this.changeTab('page2') },
        { label: '上传', icon: 'pi pi-download', command: () => this.changeTab('page3') },
        // 添加更多选项卡
      ],
      menus2: [
        { label: 'Upload', icon: 'pi pi-upload', command: () => this.changeTab('page1') },
        { label: 'Choose', icon: 'pi pi-caret-right', command: () => this.changeTab('page2') },
        { label: 'Upload', icon: 'pi pi-download', command: () => this.changeTab('page3') },
        // 添加更多选项卡
      ],


    }
  },
  methods: {
    /* 以下为UI行为调整参数
     * 1. calcWinLeftPercent: 调整桌面的宽度百分比
     * 2. resizeDesktopUI: 源码触发行为调整
     * */
    calcDesktopLeftPercent(isLeft){
      let width = this.winSize.width
      if(width <= 0){
        if(isLeft) return 20
        return 80
      }else{
        let leftPercent = Math.round((320 / width) * 100)
        let rightPercent = 100 - leftPercent
        if(isLeft) return leftPercent
        return rightPercent
      }
    },
    calcDesktopRightPercent(isTop){
      let height = this.winSize.height
      if(height <= 0){
        if(isTop) return 15
        return 85
      }else{
        let topPercent = Math.round((120 / height) * 100)
        let bottomPercent = 100 - topPercent
        if(isTop) return topPercent
        return bottomPercent
      }
    },
    resizeDesktopUI(splitName){
      try{
        if (!this.padflag && !this.phoneflag) {
          // 对于桌面，尽量保持左侧大小为320px, 但是右侧随意延展，下列为框架源代码
          let children = [...this.$refs[splitName].$el.children].filter((child) => child.getAttribute('data-pc-name') === 'splitterpanel');
          let _panelSizes = [];

          this.$refs[splitName].panels.map((panel, i) => {
            let panelInitialSize = panel.props && panel.props.size ? panel.props.size : null;
            let panelSize = panelInitialSize || 100 / this.$refs[splitName].panels.length;

            _panelSizes[i] = panelSize;
            children[i].style.flexBasis = 'calc(' + panelSize + '% - ' + (this.$refs[splitName].panels.length - 1) * this.$refs[splitName].gutterSize + 'px)';
          });

          this.$refs[splitName].panelSizes = _panelSizes;
          this.$refs[splitName].prevSize = parseFloat(_panelSizes[0]).toFixed(4);
        }
      }catch (error){
        console.log(error)
      }
    }
  },
  setup() {
    const show = ref(false);
    const showPopup = () => {
      show.value = true;
    };
    const activeNames = ref(['1']);
    const onClick = () => {
      store.commit('app/SET_AREAFLAGE',true)
      if(locale.value == 'en')
      showToast('Switching succeeded');
      else
      showToast('切换成功');
    };
    const changeLang1 = (type) => {
      locale.value = type;
      localStorage.setItem('lang',type)
      // location.reload();
    };

    // 语言类型对象读取
    const {locale} = useI18n();
    const changeTab = function(tab) {
      // 点击选项卡时更新activeTab的值
      this.activeTab = tab;
    };
    const onTabChange = function(event) {
      // 在TabMenu中的选项卡切换时更新activeTab的值
      this.activeTab = event.item.label;
    };
    return {
      show,
      showPopup,
      activeNames,
      onClick,
      changeLang1,
      locale,
      changeTab,
      onTabChange,
    };
  }
}
</script>

<style scoped>
.splitter-class {
  height: calc(100% - 0px)
}
.but{
  text-align: left;
  position: relative;
  background-image: linear-gradient(to right, var(--bluegray-500), var(--bluegray-800));
}
.but-2 {
  position: absolute;
  text-align: center;
  width: 100%;
  font-size: 36px;
  color: #FFFFFF;
}
.dialog {
  text-align: center;
}
.phonecho {
  /* height: 5%; */
  text-align: left;
  position: relative;
  /* background-image: linear-gradient(to right, var(--bluegray-500), var(--bluegray-800)); */
}
.phonecho1 {
  position: absolute;
  left: 45%;
  top: 30%;
  /* font-size:large */
  /* color: #FFFFFF; */

}
.changelang {
  height: auto;
}
</style>
