import java.io.*;
import java.net.*;

public class TalkServer
{
    public static void main(String[] args) throws Exception
    {
        ServerSocket sersock = new ServerSocket(9999);
        System.out.println("Talk Server ready for chatting");
        Socket sock = sersock.accept();

        // reading from keyboard (keyRead object)
        BufferedReader keyB
    }
}