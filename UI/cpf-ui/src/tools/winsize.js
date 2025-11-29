import store from "@/store";

const { body } = document
const WIDTH = 992

export default {
    beforeMount(){
        window.addEventListener('resize', this.resizeHandler)
    },
    beforeDestroy() {
        window.removeEventListener('resize', this.resizeHandler)
    },
    mounted() {
        this.resizeHandler()
    },

    methods: {
      resizeHandler(){
          const rect = body.getBoundingClientRect()
          let winSize = {
              height: rect.height,
              width: rect.width
          }
          store.dispatch("app/setWinSize", winSize)

        //  用户机型判断
        // 993
          if(store.state.app.winSize.width <= 1049 && store.state.app.winSize.width > 768) {
            store.commit('app/SET_PADFLAGE',true)
        } else {
            store.commit('app/SET_PADFLAGE',false)
        }

        if(store.state.app.winSize.width < 768) {
            store.commit('app/SET_PHONEFLAGE',true)
            store.commit('app/SET_SIZEFLAGE',false)
        } else {
            store.commit('app/SET_PHONEFLAGE',false)
            store.commit('app/SET_SIZEFLAGE',true)
        }
        console.log("当前的窗体大小缓存", winSize)
        console.log("当前平板标识", store.state.app.padflag)
        console.log("当前手机标识", store.state.app.phoneflag)
      }
    }
}
