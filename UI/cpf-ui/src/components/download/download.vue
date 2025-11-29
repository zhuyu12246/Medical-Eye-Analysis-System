<template>
  <div>
    <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px; margin-right: 10px">
      <div style="flex: 2;">
        <Badge :value="$t('messages.Step3')" severity="info"></Badge>
      </div>
      <div style="flex: 4">
        <Button :label="$t('messages.Download')" icon="pi pi-spin pi-spinner" @click="download" :disabled="isDownload"/>
      </div>
    </div>
  </div>
</template>

<script>
import Badge from 'primevue/badge'
import ProgressBar from 'primevue/progressbar';
import Button from 'primevue/button';
import {mapActions, mapState} from "vuex";
import api from '@/api'

export default {
  name: "download",
  components: {
    Button,
    ProgressBar,
    Badge
  },
  computed: {
    ...mapState("task", ['taskId']),
    ...mapState("app", ['isDownload'])
  },
  methods: {
    ...mapActions({
      setMsg: 'message/setMsg',
      setError: 'error/setErrorInfo'
    }),
    download(){
      const url = api.TASK_DOWNLOAD
      const query = url + "?query_key=" + this.taskId
      console.log(query)
      if(this.taskId === ""){
        this.setError("Create Task first")
      }else{
        const a = document.createElement('a')
        a.href = query
        a.style.display = 'none'
        a.target = "_blank"
        document.body.appendChild(a)
        a.click()
        setTimeout(()=>{
          document.body.removeChild(a)
        }, 5 * 60 * 1000)
      }
    }
  }
}
</script>

<style scoped>

</style>
