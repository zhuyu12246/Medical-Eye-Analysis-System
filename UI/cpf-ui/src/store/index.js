import { createStore } from 'vuex'
import app from './modules/app'
import user from "./modules/user"
import error from "./modules/error";
import task from "./modules/task"
import message from "@/store/modules/message";
import file from './modules/file'

const store = createStore({
  modules: {
    app,
    user,
    error,
    task,
    message,
    file
  }
})


export default store
