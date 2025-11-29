<template>
  <div>
    <Toast />
  </div>
</template>

<script>
import Toast from "primevue/toast";
import {useToast} from "primevue/usetoast";
import {mapState} from "vuex";

export default {
  name: "toast-helper",
  components: {
    Toast
  },
  mounted() {
    this.toastObj = useToast()
  },
  data(){
    return {
      toastObj: null
    }
  },
  computed: {
    ...mapState("message", ['msg', 'type']),
    ...mapState("error", ['errorInfo'])
  },
  watch: {
    msg(newVal, oldVal){
      if (this.type === 'success'){
        this.showSuccess(newVal)
      }else if(this.type === "info"){
        this.showInfo(newVal)
      }else if(this.type === 'warn'){
        this.showWarning(newVal)
      }
    },
    errorInfo(newVal, oldVal){
      this.showError(newVal)
    }
  },
  methods: {
    showSuccess(detail){
      this.toastObj.add({ severity: 'success', summary: "Success", detail: detail, life: 5000 });
    },
    showInfo(detail){
      this.toastObj.add({ severity: 'info', summary: 'Info', detail: detail, life: 3000 });
    },
    showWarning(detail){
      this.toastObj.add({ severity: 'warn', summary: 'Warn', detail: detail, life: 8000 });
    },
    showError(detail){
      this.toastObj.add({ severity: 'error', summary: 'Error', detail: detail, life: 15000 });
    }
  }
}
</script>

<style scoped>

</style>
