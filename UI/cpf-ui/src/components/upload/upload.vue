<template>
  <div>
    <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px; margin-right: 10px">
      <div style="flex: 2">
        <Badge :value="$t('messages.Step1')" severity="info"></Badge>
      </div>
      <div style="flex: 4">
        <Button :label="$t('messages.Submit')" icon="pi pi-check" @click="visible = true" />
      </div>
    </div>

    <Dialog v-model:visible="visible" modal :header="$t('messages.UploadPanel')" :style="{ width: '50rem'}" :breakpoints="{ '1199px': '75vw', '575px': '90vw' }">
      <FileUpload
          :chooseLabel="$t('messages.Choose')"
          :uploadLabel="$t('messages.Upload')"
          :cancelLabel="$t('messages.Cancel')"
          :multiple="true"
          accept="image/*"
          :maxFileSize="15000000"
          :fileLimit="30"
          customUpload
          @uploader="customUpload"
          ref="uploader"
      >
        <template #empty>
          <p v-if="uoloadFlag">{{ $t('messages.Draganddropfilestoheretoupload') }}</p>
          <ul class="uploaded-files-list">
            <li v-for="file in uploadedFiles" :key="file.name">
              {{ file.name }}
              <span>Completed</span>
            </li>
          </ul>
        </template>

      </FileUpload>
    </Dialog>
  </div>
</template>

<script>
import Badge from 'primevue/badge'
import Button from 'primevue/button';
import Dialog from 'primevue/dialog';
import FileUpload from 'primevue/fileupload';
import uploadhelper from "@/components/upload/uploadhelper";
import task from "@/store/modules/task";
import store from "@/store";

export default {
  name: "upload",
  mixins: [uploadhelper],
  components: {
    Button,
    Dialog,
    FileUpload,
    Badge
  },
  data(){
    return {
      visible: false
    }
  },
  methods: {
    customUpload(event){
      if(this.uploadedFiles != []){
        this.uploadedFiles = []
        this.setList([])
      }
      store.commit('app/SET_ISRUNNING',false)
      const task_id = this.createTaskId()
      if (task_id !== null){
        console.log(task_id)
        this.uploadFileList(this.$refs.uploader, task_id, event.files)
      }
    },
  }
}
</script>

<style scoped>
  .uploaded-files-list {
    list-style-type: none;
    padding: 0;
}

.uploaded-files-list li {
    background-color: #f5f5f5;
    border: 1px solid #ccc;
    margin-bottom: 5px;
    padding: 10px;
    color: #333;
    position: relative; /* 设置相对定位 */
}

.uploaded-files-list li span {
  position: absolute; /* 设置绝对定位 */
    top: 50%;
    right: 10px;
    transform: translateY(-50%);
    background-color: #42b983;
    color: white;
    padding: 3px 8px;
    border-radius: 20px; /* 圆角边框使其呈现胶囊状 */
}
</style>
