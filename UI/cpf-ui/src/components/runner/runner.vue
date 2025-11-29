<template>
  <div>
    <!-- cascade select   -->
    <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px; margin-right: 10px; margin-left: 1px">
      <div style="flex: 2;">
        <Button severity="warning" :label="$t('messages.Enh')" style="font-size: 12px; color: #333333"/>
      </div>
      <div style="flex: 4">
        <SelectButton  :options="enhOptions" v-model="enhValue"></SelectButton>
      </div>
    </div>

    <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px; margin-right: 10px; margin-left: 1px">
      <div style="flex: 3;">
        <Button severity="warning" :label="$t('messages.Seg')" style="font-size: 12px; color: #333333"/>
      </div>
      <div style="flex: 6">
          <TreeSelect v-model="segValue" :options="segOptions" placeholder="Select Item" class="md:w-20rem w-full" />
      </div>
    </div>

    <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px; margin-right: 10px; margin-left: 1px">
      <div style="flex: 2;">
        <Button severity="warning" :label="$t('messages.Grd')" style="font-size: 12px; color: #333333"/>
      </div>
      <div style="flex: 4">
        <SelectButton :options="gradeOptions" v-model="gradeValue" disabled></SelectButton>
      </div>
    </div>

    <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px; margin-right: 10px">
      <div style="flex: 2;">
        <Badge :value="$t('messages.Step2')" severity="info"></Badge>
      </div>
      <div style="flex: 4">
        <Button :label="$t('messages.Running')" icon="pi pi-spin pi-cog" @click="startToRun" :disabled="isRunning"/>
      </div>
    </div>
    <div style="margin: 20px 10px 0px 10px">
      <Knob v-model="frozenValue" :size="150"></Knob>
    </div>
  </div>
</template>

<script>
import Badge from 'primevue/badge'
import ProgressBar from 'primevue/progressbar';
import Button from 'primevue/button';
import Knob from 'primevue/knob'
import {mapActions, mapState} from "vuex";
import store from "@/store";
import api from '@/api'
import runnerHelper from "@/components/runner/runnerHelper";
import InputGroup from 'primevue/inputgroup';
import InputGroupAddon from 'primevue/inputgroupaddon';
import SelectButton from 'primevue/selectbutton';
import TreeSelect from 'primevue/treeselect';



export default {
  name: "runner",
  components: {
    Button,
    ProgressBar,
    Badge,
    Knob,
    InputGroup,
    InputGroupAddon,
    SelectButton,
    TreeSelect
  },
  mixins: [runnerHelper],
  methods: {
    ...mapActions({
      setRunningOptions: 'task/setTaskOptions'
    }),
    startToRun(){
      // console.log(this.uploadValue)
      store.commit('app/SET_DOWNLOAD',false)
      this.setRunningOptions(this.uploadValue)
      this.listenLoop()
    }
  },
  watch: {
    enhValue(newValue, oldValue){
      if (newValue === 'on'){
        this.uploadValue['enhance'] = 0
      }else{
        this.uploadValue['enhance'] = 1
      }
    },
    segValue(newValue, oldValue){
      if(newValue.hasOwnProperty(0)){
        this.uploadValue['seg'] = 0
      }else{
        this.uploadValue['seg'] = 1
      }
    }
  },
  data(){
    return {
      enhValue: "on",
      enhOptions: ['on', 'off'],
      segValue: {1: true},
      segOptions: [
        {
          key: '0',
          label: 'camw',
          data: 0,
          icon: 'pi pi-fw pi-cog'
        },
        {
          key: '1',
          label: 'unet',
          data: 1,
          icon: 'pi pi-fw pi-cog'
        }
      ],
      gradeValue: 0,
      gradeOptions: ['on', 'off'],
      uploadValue: {
        'enhance': 0,
        'seg': 0,
        'grade': 0
      },
      value:0
    }
  },
  computed: {
    frozenValue() {
      return Object.freeze(this.value);
    },
    ...mapState("app", ['isRunning'])
  }
}
</script>

<style scoped>

</style>
