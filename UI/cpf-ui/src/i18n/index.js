import { createI18n } from "vue-i18n"
// 英文翻译包
import en from './langs/en'
// 中文翻译包
import zh from './langs/zh'

const messages = {
    zh,
    en
}
const i18n = createI18n({
    legacy:false,
    locale:localStorage.getItem('lang') || 'zh',
    messages
})

export default i18n
