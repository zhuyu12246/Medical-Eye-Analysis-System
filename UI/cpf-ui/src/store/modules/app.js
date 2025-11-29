
const state = {
    winSize: {
        height: 1080,
        width: 1920
    },
    // pad判断
    padflag: false,
    // phone判断
    phoneflag: false,
    runningStep: 0,
    // 控制table-area显示个数
    areaflage:false,
    sizeflage:true,
    // 是否关闭running按钮
    isRunning:true,
    // 是否关闭download按钮
    isDownload:true,
}

const mutations = {
    SET_WIN_SIZE: (state, size) => {
        state.winSize = size
    },
    SET_RUNNING_STEP: (state, step) => {
        state.runningStep = step
    },
    // pad修改
    SET_PADFLAGE: (state, flage) => {
        state.padflag = flage
    },
     // phone修改
    SET_PHONEFLAGE: (state, flage) => {
        state.phoneflag = flage
    },
    SET_AREAFLAGE: (state) => {
        state.areaflage = !state.areaflage
    },
    SET_SIZEFLAGE: (state, flage) => {
        state.sizeflage = flage
    },
    SET_ISRUNNING: (state, flage) => {
        state.isRunning = flage
    },
    SET_DOWNLOAD: (state, flage) => {
        state.isDownload = flage
    },

}

const actions = {
    setWinSize({ commit }, size) {
        commit('SET_WIN_SIZE', size)
    },
    setRunningStep({ commit }, step){
        commit('SET_RUNNING_STEP', step)
    }
}

export default {
    namespaced: true,
    state,
    mutations,
    actions
}
