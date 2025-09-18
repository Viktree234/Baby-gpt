import makeWASocket, { useMultiFileAuthState, DisconnectReason } from "@whiskeysockets/baileys"
import fetch from "node-fetch"
import express from "express"
import { createServer } from "http"
import { Server } from "socket.io"
import qrcode from "qrcode"

const app = express()
const server = createServer(app)
const io = new Server(server)

let latestQR = ""

async function connectBot() {
    const { state, saveCreds } = await useMultiFileAuthState("session")

    const sock = makeWASocket({
        auth: state,
        printQRInTerminal: false,
    })

    sock.ev.on("creds.update", saveCreds)

    sock.ev.on("connection.update", (update) => {
        const { connection, lastDisconnect, qr } = update

        if (qr) {
            qrcode.toDataURL(qr, (err, url) => {
                if (!err) {
                    latestQR = url
                    console.log("📲 New QR generated")
                    io.emit("qr", latestQR) // broadcast to all connected clients
                }
            })
        }

        if (connection === "close") {
            const reason = lastDisconnect?.error?.output?.statusCode
            if (reason !== DisconnectReason.loggedOut) {
                connectBot()
            }
        } else if (connection === "open") {
            console.log("✅ WhatsApp bot connected!")
            latestQR = ""
            io.emit("ready") // notify web clients that login is done
        }
    })

    sock.ev.on("messages.upsert", async (msg) => {
        const m = msg.messages[0]
        if (!m.message || m.key.fromMe) return

        const from = m.key.remoteJid
        const text =
            m.message.conversation ||
            m.message.extendedTextMessage?.text ||
            ""

        console.log("📩 Received:", text)

        if (text) {
            try {
                const res = await fetch("http://127.0.0.1:5000/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: text }),
                })
                const data = await res.json()
                const reply = data.reply || "🤖 I don’t know what to say yet."

                await sock.sendMessage(from, { text: reply })
                console.log("🤖 Replied:", reply)
            } catch (err) {
                console.error("❌ Error contacting AI model:", err)
            }
        }
    })
}

// Serve the QR page
app.get("/", (req, res) => {
    res.send(`
        <html>
            <head>
                <title>WhatsApp Login</title>
                <script src="/socket.io/socket.io.js"></script>
                <script>
                    const socket = io();
                    socket.on("qr", (qr) => {
                        document.getElementById("qr").src = qr;
                        document.getElementById("status").innerText = "📱 Scan QR to login";
                    });
                    socket.on("ready", () => {
                        document.getElementById("qr").style.display = "none";
                        document.getElementById("status").innerText = "✅ WhatsApp Connected!";
                    });
                </script>
            </head>
            <body style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;">
                <h2 id="status">Waiting for QR...</h2>
                <img id="qr" style="width:300px;height:300px;" />
            </body>
        </html>
    `)
})

server.listen(3000, () => console.log("🌐 QR Web running at http://localhost:3000"))

connectBot()
