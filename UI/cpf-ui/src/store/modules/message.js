
const state = {
    msg: "",
    type: ""
}

const mutations = {
    SET_MSG: (state, info) => {
        state.msg = info.msg
        state.type = info.type
    }
}

const actions = {
    setMsg({ commit }, info) {
        commit('SET_MSG', info)
    }
}

export default {
    namespaced: true,
    state,
    mutations,
    actions
}
