import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import mongoData from "./mongoData.js"
import Pusher from "pusher";

// app config
const app=express()
const port = process.env.PORT || 8002

const pusher = new Pusher({
    appId: "appId",
    key: "key",
    secret: "secret",
    cluster: "cluster",
    useTLS: true
  });

// middlewares
app.use(express.json())
app.use(cors())

// db config
const mongoURI = 'mongodb+srv://admin:<password>@cluster0.gcy2f.mongodb.net/discordDB?retryWrites=true&w=majority'
//password SU2uZY3qrhPFqMuS
mongoose.connect(mongoURI, {

    useNewUrlParser: true,
    useUnifiedTopology: true
})

mongoose.connection.once('open', () => {
    console.log('DB Connected')

    const changeStream = mongoose.connection.collection('conversations').watch()

    changeStream.on('change', (change) => {
        if (change.operationType === 'insert') {
            pusher.trigger('channels', 'newChannel', {
                'change': change
            });
        } else if (change.operationType === 'update') {
            pusher.trigger('conversations', 'newMessage', {
                'change': change
            });
        } else {
            console.log('Error triggering Pusher')
        }
    })
})

// api routes
app.get('/', (req, res) => res.status(200).send('hello world'))

app.post('/new/channel', (req, res)=>{
    const dbData = req.body

    mongoData.create(dbData, (err, data) => {
        if(err) {
            res.status(500).send(err)
        } else {
            res.status(201).send(data)
        }
    })
})

app.get('/get/channelList', (req, res)=>{
    mongoData.find((err, data)=>{
        if(err) {
            res.status(500).send(err)
        } else {
            let channels = []

            data.map((channelData)=>{
                const channelInfo = {
                    id: channelData._id,
                    name: channelData.channelName
                }
                channels.push(channelInfo)
            })

            res.status(200).send(channels)
        }
    })
})

app.post('/new/message', (req, res)=>{
    const newMessage = req.body

    mongoData.updateMany(
        {_id: req.query.id},
        {$push: {conversation: req.body}},
        (err, data)=>{
            if (err) {
                console.log('Error saving message...')
                console.log(err)

                res.status(500).send(err)
            } else {
                res.status(201).send(data)
            }
        }
    )
})

app.get('/get/data', (req,res) => {
    mongoData.find((err, data) => {
        if(err) {
            res.status(500).send(err)
        } else {
            res.status(200).send(data)
        }
    })
})

app.get('/get/conversation', (req, res) => {
    const id = req.query.id

    mongoData.find({ _id: id}, (err, data) => {
        if (err) {
            res.status(500).send(err)
        } else {
            res.status(200).send(data)
        }
    })
})

// listen
app.listen(port, ()=>console.log(`listening on localhost:${port}`))