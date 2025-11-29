// const BASE_URL = "http://192.168.3.99:9000"
const BASE_URL = "http://114.55.245.149:8051"
//const BASE_URL = "http://114.55.245.149:8050"

export default {
    TASK_GEN: BASE_URL + "/gateway/task/generate",
    TASK_POST: BASE_URL + "/gateway/task/post",
    TASK_IN: BASE_URL + "/gateway/task/in",
    TASK_QUERY: BASE_URL + "/gateway/task/query",
    TASK_DOWNLOAD: BASE_URL + "/gateway/task/download",
    RESULT_QUERY: BASE_URL + "/gateway/result/query"
}
