import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import PrimeVue from "primevue/config";
import ToastService from 'primevue/toastservice';
import { Popup, Icon, Collapse, CollapseItem, FloatingBubble, Button} from 'vant';
import 'vant/lib/index.css';
import i18n from './i18n';

const app = createApp(App)
app.use(Popup)
app.use(Icon)
app.use(Collapse)
app.use(CollapseItem)
app.use(FloatingBubble)
app.use(Button)

app.use(store)
app.use(router)
app.use(PrimeVue)
app.use(ToastService)

app.use(i18n)

app.mount('#app')


