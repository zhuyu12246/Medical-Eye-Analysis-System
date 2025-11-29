<template>
  <div style="height: 100%">
<!--    <div v-if="show" style="height: 100%">-->
<!--      <framework></framework>-->
<!--    </div>-->
    <framework></framework>
  </div>
</template>

<script>
import framework from "@/components/framework/framework";
import {mapState} from "vuex";

export default {
  name: 'HomeView',
  components: {
    framework
  },
  computed:{
    ...mapState("app", ['winSize'])
  },
  watch: {
    winSize(newSize, oldSize){
      this.show = false
      setTimeout(()=>{
        this.show = true
      }, 100)
    }
  },
  data(){
    return {
      show: true
    }
  }
}
</script>
