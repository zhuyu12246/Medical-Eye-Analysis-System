
const state = {
    errorInfo: ""
}

const mutations = {
    SET_ERROR_INFO: (state, info) => {
        state.errorInfo = info
    }
}

const actions = {
    setErrorInfo({ commit }, info) {
        commit('SET_ERROR_INFO', info)
    }
}

export default {
    namespaced: true,
    state,
    mutations,
    actions
}
