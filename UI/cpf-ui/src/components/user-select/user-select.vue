<template>
  <div style="margin: 20px 3px 0px 3px;cursor:pointer">
    <Listbox
        :emptyMessage="$t('messages.Noavailableoptions')"
        v-model="selectedName" :options="names"
        optionLabel="name" class="w-full md:w-14rem"
        :listStyle="getHeight"
    />
  </div>
</template>

<script>
import Listbox from 'primevue/listbox';
import {mapActions, mapState} from "vuex";
export default {
  name: "user-select",
  components: {
    Listbox
  },
  computed: {
    ...mapState('app', ['winSize']),
    ...mapState('file', ['uploadList', 'uploadChange']),
    getHeight(){
      let line = this.height * 0.30 + 'px'
      return `max-height: ${line}`
    }
  },
  watch: {
    winSize(newVale, oldValue){
      this.height = newVale.height
    },
    selectedName(newValue, oldValue){
      this.setFile(newValue.name)
    },
    uploadChange(newValue, oldValue){
      this.names = this.uploadList
    }
  },
  methods: {
    ...mapActions({
      setFile: 'file/setSelectName'
    })
  },
  data() {
    return {
      height: 1080,
      selectedName: "",
      names: [

      ]
    }
  }
}
</script>

<style scoped>

</style>
