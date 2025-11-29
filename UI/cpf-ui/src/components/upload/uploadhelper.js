import {mapActions, mapState} from "vuex";
import api from "@/api"
import { showDialog, showToast, showLoadingToast,showSuccessToast } from 'vant';

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }
function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }
function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }
function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }
function _createForOfIteratorHelper(o, allowArrayLike) { var it = typeof Symbol !== "undefined" && o[Symbol.iterator] || o["@@iterator"]; if (!it) { if (Array.isArray(o) || (it = _unsupportedIterableToArray(o)) || allowArrayLike && o && typeof o.length === "number") { if (it) o = it; var i = 0; var F = function F() {}; return { s: F, n: function n() { if (i >= o.length) return { done: true }; return { done: false, value: o[i++] }; }, e: function e(_e) { throw _e; }, f: F }; } throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); } var normalCompletion = true, didErr = false, err; return { s: function s() { it = it.call(o); }, n: function n() { var step = it.next(); normalCompletion = step.done; return step; }, e: function e(_e2) { didErr = true; err = _e2; }, f: function f() { try { if (!normalCompletion && it["return"] != null) it["return"](); } finally { if (didErr) throw err; } } }; }
function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }
function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) arr2[i] = arr[i]; return arr2; }

export default {
    data(){
      return {
        progress: 0,
        uploadQueue: [], // 文件上传队列
        isUploading: false, // 标志以跟踪上传是否正在进行中
        uploadedFiles: [],
        uoloadFlag:true

      }
    },
    computed: {
        ...mapState('user', ['userInfo']),
        ...mapState('file', ['uploadChange','uploadList'])
    },
    methods: {
        ...mapActions({
            setErrorInfo: 'error/setErrorInfo',
            setTaskId: 'task/setTaskId',
            setStep: 'app/setRunningStep',
            setList: 'file/setUploadList',
            addList: 'file/addUploadList',
            setChange: 'file/setUploadChang',
        }),
        createTaskId(){
            const xhr = new XMLHttpRequest();
            const url = api.TASK_GEN
            const name = this.userInfo.username
            const query = url + "?user=" + name

            try{
                xhr.open("GET", query, false)
                xhr.send()

                const response = xhr.responseText
                const jsonResult = JSON.parse(response)
                const remoteRes = jsonResult['res']
                if (xhr.status === 200){
                    this.setTaskId(remoteRes)
                    return remoteRes
                }else{
                    this.setErrorInfo(remoteRes)
                    return null
                }
            }catch (error) {
                this.setErrorInfo("Server un-catch error")
                return null
            }
        },
        // 上传单个文件
        uploadFile(uploadControl, queryKey, file) {
            const xhr = new XMLHttpRequest();
            let formData = new FormData();
            
            // 获取没有后缀的文件名
            let fileNameWithoutExtension = file.name.replace(/\.[^/.]+$/, '');

            // 替换文件名中的特殊字符
            let sanitizedFileName = fileNameWithoutExtension.replace(/[\\/:*?"<>|+~:@!#$%&.]/g, '_');

            // 获取文件的后缀并与处理后的文件名拼接
            let fileExtension = file.name.substring(file.name.lastIndexOf('.'));
            let finalFileName = sanitizedFileName + fileExtension;

            // console.log(finalFileName)

            // 将文件附加到 FormData 中，使用处理后的文件名
            formData.append("files", file, finalFileName);

            // query key
            formData.append('query_key', queryKey);

            // progress
            const _this = this;
            xhr.upload.addEventListener('progress', function (event) {
                if (event.lengthComputable) {
                 _this.progress = Math.round(event.loaded * 100 / event.total);
                }
                uploadControl.$emit('progress', {
                    originalEvent: event,
                    progress: _this.progress
                });
            });

            // onready
            xhr.onreadystatechange = function () {
                if (xhr.readyState === 4) {
                    _this.progress = 0;
                    if (xhr.status >= 200 && xhr.status < 300) {
                        if (uploadControl.fileLimit) {
                            uploadControl.uploadedFileCount++;
                        }
                        uploadControl.$emit('upload', {
                            xhr: xhr,
                            files: [file]
                        });
                        _this.uploadedFiles.push({ name: finalFileName });
                        _this.setStep(1);
                        // _this.setList([{ name: finalFileName, code: finalFileName }]);
                        _this.addList({ name: finalFileName, code: finalFileName });
                        _this.setChange(!_this.uploadChange);
                    } else {
                        uploadControl.$emit('error', {
                            xhr: xhr,
                            files: [file]
                        });
                    }

                    uploadControl.clear();

                    // 设置 isUploading 为 false 以表示上传完成
                    _this.isUploading = false;

                    // 上传队列中的下一个文件
                    _this.uploadNextFile(uploadControl, queryKey);
                }
            };
            // post
            const url = api.TASK_POST;
            xhr.open('POST', url, true);
            xhr.send(formData);
        },

        // 上传队列中的下一个文件
        uploadNextFile(uploadControl, queryKey) {
            // showToast('上传中ing....');
            showLoadingToast({
                message: 'uploading...',
                forbidClick: true,
                duration:0
              });
            if (this.uploadQueue.length > 0 && !this.isUploading) {
                // 如果队列中有文件且没有上传正在进行中，则开始上传下一个文件
                const nextFile = this.uploadQueue.shift();
                this.isUploading = true;
                this.uploadFile(uploadControl, queryKey, nextFile);
            } else {
                showSuccessToast('done!');
                // showToast('上传成功');
            }
        },

        // 上传文件列表，添加到队列中
        uploadFileList(uploadControl, queryKey, fileList) {
            this.uoloadFlag = false;
            // 将文件添加到队列中
            this.uploadQueue.push(...fileList);

            // 如果没有上传正在进行中，则开始上传下一个文件
            this.uploadNextFile(uploadControl, queryKey);
        },

    }
}
