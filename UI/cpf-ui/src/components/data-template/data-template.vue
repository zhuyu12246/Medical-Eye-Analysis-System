<template>
  <div style="height: 100%; width: 100%">
    <div style="height: 100%; width: 100%; overflow: auto">
      <h4 v-if="ori">{{ $t('messages.OriData') }}</h4>
      <h4 v-else>{{ $t('messages.EnhData') }}</h4>

      <div style="text-align: center" v-if="noValue">
        {{ $t('messages.NoSource') }}
      </div>
      <table style="width: 100%" v-else border="1">
        <tr>
          <td>{{ $t('messages.Name') }}</td>
          <td>{{ $t('messages.Result') }}</td>
        </tr>
        <tr>
          <td>{{ $t('messages.file') }}</td>
          <td>
            <div style="display: flex; align-items: center; justify-content: center">
              <div v-if="file === ''" style="width: 250px; height: 250px;border: 1px gray solid"></div>
              <Image v-else :src="file" alt="Image" width="250" preview />
            </div>
          </td>
        </tr>
        <tr>
          <td>{{ $t('messages.seg1') }}</td>
          <td>
            <div style="display: flex; align-items: center; justify-content: center">
              <div v-if="seg === ''" style="width: 250px; height: 250px;border: 1px gray solid"></div>
              <Image v-else :src="seg" alt="Image" width="250" preview />
            </div>
          </td>
        </tr>
        <tr>
          <td>{{ $t('messages.cam') }}</td>
          <td>
            <div style="display: flex; align-items: center; justify-content: center">
              <div v-if="cam === ''" style="width: 250px; height: 250px;border: 1px gray solid"></div>
              <Image v-else :src="cam" alt="Image" width="250" preview />
            </div>
          </td>
        </tr>
        <tr>
          <td>{{ $t('messages.camj') }}</td>
          <td>
            <div style="display: flex; align-items: center; justify-content: center">
              <div v-if="camPlus === ''" style="width: 250px; height: 250px;border: 1px gray solid"></div>
              <Image v-else :src="camPlus" alt="Image" width="250" preview />
            </div>
          </td>
        </tr>
        <tr>
          <td>grade</td>
          <td>
            <Chart type="pie" :data="chartData" :options="chartOptions" class="w-full md:w-30rem"
                   style="width: 320px; height: 320px; margin: auto"
            />
          </td>
        </tr>
      </table>
    </div>
  </div>
</template>

<script>
import api from '@/api';
import {mapActions, mapState} from "vuex";
import Image from 'primevue/image';
import Chart from 'primevue/chart';

export default {
  name: "data-template",
  components: {
    Image,
    Chart
  },
  props: {
    ori: {
      type: Boolean,
      default: true
    }
  },
  computed: {
    ...mapState('file', ['selectName', 'remoteJsonStr']),
    ...mapState('task', ['taskId', 'taskOptions'])
  },
  watch: {
    // TODO: xhr异步修改属性，导致浏览器debug跟踪丢失
    selectName(newValue, oldValue){
      if(this.ori){
        this.runningLoop()
      }
    },
    remoteJsonStr(newValue, oldValue){
      if(this.ori === false){
        let _ = this.parserJson(newValue, false)
      }
    }
  },
  data(){
    return {
      file: "",
      seg: '',
      cam: '',
      camPlus: '',
      chartData: {},
      chartOptions: {},
      noValue: false
    }
  },
  mounted() {
    this.chartOptions = this.setChartOptions()
    this.chartData = this.setChartData()
  },
  methods: {
    ...mapActions({
      setJsonStr: 'file/setRemoteJsonStr',
      setError: 'error/setErrorInfo'
    }),
    setChartData(){
      const documentStyle = getComputedStyle(document.body);

      return {
        labels: ['0', '1', '2', '3', '4'],
        datasets: [
          {
            data: [100, 100, 100, 100, 100],
            backgroundColor: [documentStyle.getPropertyValue('--blue-500'), documentStyle.getPropertyValue('--yellow-500'), documentStyle.getPropertyValue('--green-500'), documentStyle.getPropertyValue('--purple-500'), documentStyle.getPropertyValue('--gray-500')],
            hoverBackgroundColor: [documentStyle.getPropertyValue('--blue-400'), documentStyle.getPropertyValue('--yellow-400'), documentStyle.getPropertyValue('--green-400'), documentStyle.getPropertyValue('--purple-400'), documentStyle.getPropertyValue('--gray-400')]
          }
        ]
      };
    },
    setChartOptions() {
      const documentStyle = getComputedStyle(document.documentElement);
      const textColor = documentStyle.getPropertyValue('--text-color');

      return {
        plugins: {
          legend: {
            labels: {
              usePointStyle: true,
              color: textColor
            }
          }
        }
      }
    },
    runningLoop(){
      var interval = setInterval(()=> {
        if(this.taskId !== "" && this.selectName !== ""){
          const json_str = JSON.stringify(this.taskOptions)
          const queryKey = api.RESULT_QUERY + "?query_key=" + this.taskId + "&file_name=" + this.selectName + "&running_options=" + json_str
          const xhr = new XMLHttpRequest()
          try {
            xhr.open("GET", queryKey, false)
            xhr.send()

            const res = xhr.responseText
            this.setJsonStr(res)
            let done = this.parserJson(res, true)
            if(done) {
              clearInterval(interval)
            }
          }catch (error){
            this.setError('Server un-catch error')
          }
        }
      }, 1500)
    },
    parserJson(jsonStr, ori){
      if(!ori){
        if(this.taskOptions['enhance'] === 1) {
          this.noValue = true
          return true
        }
      }


      let res = JSON.parse(jsonStr)['res']
      let json = JSON.parse(res)

      let d = {}
      if (ori === true) d = json['ori']
      else d = json['enh']

      this.file = d['file']
      this.seg = d['seg']
      this.cam = d['grade']['cam']
      this.camPlus = d['grade']['cam_plus']
      this.chartData['datasets'][0].data = d['grade']['pie']
      return json['done']
    }
  }
}
</script>

<style scoped>

</style>
