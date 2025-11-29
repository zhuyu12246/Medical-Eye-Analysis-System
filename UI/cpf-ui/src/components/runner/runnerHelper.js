import {mapActions, mapState} from "vuex";
import api from "@/api"

export default {
    data(){
        return {
            value: 0,
            intervalStop: false
        }
    },
    computed: {
        ...mapState("task", ['taskId', 'taskOptions'])
    },
    methods: {
        ...mapActions({
            setStep: 'app/setRunningStep',
            setErrorInfo: 'error/setErrorInfo',
            setMsg: 'message/setMsg'
        }),
        putTaskToRemote(){
            const xhr = new XMLHttpRequest()
            const url = api.TASK_IN

            if(this.taskId === ""){
                this.setErrorInfo("Create task first")
                return -1
            }

            const json_str = JSON.stringify(this.taskOptions)
            const query = url + "?query_key=" + this.taskId + "&running_options=" + json_str

            try{
                xhr.open("PUT", query, false)
                xhr.send()

                const response = xhr.responseText
                const jsonResult = JSON.parse(response)
                const remoteRes = jsonResult['res']
                if (xhr.status === 200){
                    return 0
                }else{
                    this.setErrorInfo(remoteRes)
                    return -1
                }
            }catch (error) {
                this.setErrorInfo("Server un-catch error")
                return -1
            }
        },
        loopFunction(){
            const xhr = new XMLHttpRequest()
            const url = api.TASK_QUERY
            const query = url + "?query_key=" + this.taskId
            try{
                xhr.open("GET", query, false)
                xhr.send()

                const response = xhr.responseText
                const jsonResult = JSON.parse(response)
                const remoteStatus = jsonResult['status']
                const remoteMsg = jsonResult['msg']
                const remoteProgress = jsonResult['progress']
                console.log(remoteStatus, remoteProgress, remoteMsg)
                if (xhr.status === 200){
                    this.value = remoteProgress
                    if (remoteStatus !== 500){
                        if(remoteMsg !== ""){
                            this.setMsg({
                                'msg': remoteMsg,
                                'type': 'info'
                            })
                        }
                        if (remoteStatus === 200){
                            this.intervalStop = true
                            this.setStep(3)
                        }
                    }else{
                        this.setErrorInfo(remoteMsg)
                        this.intervalStop = true
                    }
                }
            }catch (error) {
                this.setErrorInfo("Server un-catch error")
            }
        },

        listenLoop(){
            this.intervalStop = false
            this.value = 0
            const that = this

            let res = this.putTaskToRemote()
            if (res === 0){
                that.setMsg({
                    'msg': 'Task Upload, waiting about 30s for each file',
                    'type': 'success'
                })
                that.setStep(2)
                const interval = setInterval(function (){
                    that.loopFunction()
                    if(that.intervalStop === true){
                        clearInterval(interval)
                    }
                }, 1500)
            }
        }
    }
}
