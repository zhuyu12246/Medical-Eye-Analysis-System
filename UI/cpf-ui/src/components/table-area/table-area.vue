<template>
  <div style="height: 100%">
    <Splitter class="mb-5 splitter-class">
      <SplitterPanel class="flex align-items-center justify-content-center" v-show="phoneflag && !areaflage || sizeflage">
        <data-template :ori="true">
        </data-template>
      </SplitterPanel>
      <SplitterPanel class="flex align-items-center justify-content-center" v-show="phoneflag && areaflage || sizeflage">
        <data-template :ori="false">
        </data-template>
      </SplitterPanel>
    </Splitter>
  </div>
</template>

<script>
import Splitter from 'primevue/splitter';
import SplitterPanel from 'primevue/splitterpanel';
import Chart from 'primevue/chart';
import DataTemplate from "@/components/data-template/data-template";
import { mapState } from 'vuex'
import store from "@/store";
export default {
  name: "table-area",
  components: {
    DataTemplate,
    Chart,
    Splitter,
    SplitterPanel
  },
  computed: {
    ...mapState("app", ['padflag','phoneflag','areaflage','sizeflage'])
  },
  mounted() {
  },
  data(){
    return {

    }
  }
}
</script>

<style scoped>
.splitter-class {
  height: calc(100% - 0px)
}
</style>
