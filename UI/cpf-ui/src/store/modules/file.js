
const state = {
    uploadChange: true,
    uploadList: [],
    selectName: "",
    remoteJsonStr: "",
}

const mutations = {
    SET_UPLOAD_LIST: (state, uploadList) => {
        state.uploadList = uploadList
    },
    // 增加uploadList
    ADD_UPLOAD_LIST: (state, upload) => {
        state.uploadList.push(upload)
    },
    SET_UPLOAD_CHANGE: (state, change) => {
        state.uploadChange = change
    },
    SET_SELECT_NAME: (state, name) => {
        state.selectName = name
    },
    SET_REMOTE_JSON_STR: (state, jsonStr) => {
        state.remoteJsonStr = jsonStr
    }
}

const actions = {
    setUploadList({ commit }, uploadList) {
        commit('SET_UPLOAD_LIST', uploadList)
    },
    addUploadList({ commit }, upload) {
        commit('ADD_UPLOAD_LIST', upload)
    },
    setSelectName({ commit }, name){
        commit('SET_SELECT_NAME', name)
    },
    setUploadChang({ commit }, change){
        commit('SET_UPLOAD_CHANGE', change)
    },
    setRemoteJsonStr({ commit }, jsonStr){
        commit('SET_REMOTE_JSON_STR', jsonStr)
    }
}

export default {
    namespaced: true,
    state,
    mutations,
    actions
}
