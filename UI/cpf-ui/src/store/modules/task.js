
const state = {
    taskId: "",
    taskOptions: {
        'enhance': 0,
        'seg': 0,
        'grade': 0
    }
}

const mutations = {
    SET_TASK_ID: (state, id) => {
        state.taskId = id
    },
    SET_TASK_OPTIONS: (state, option) => {
        state.taskOptions = option
    }
}

const actions = {
    setTaskId({ commit }, id) {
        commit('SET_TASK_ID', id)
    },
    setTaskOptions({ commit }, option) {
        commit('SET_TASK_OPTIONS', option)
    }
}

export default {
    namespaced: true,
    state,
    mutations,
    actions
}
