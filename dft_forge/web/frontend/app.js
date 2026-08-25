import { initChat } from "./chat.js";
import { initViewer } from "./viewer.js";
import { initSettings } from "./settings.js";

initChat();
initSettings();
setTimeout(initViewer, 0);
