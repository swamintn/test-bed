/*
 * Downloads pdf files from CSE 548 course to given dir
 *
 */

#include <stdio.h>
#include <curl/curl.h>

#include <iostream>
#include <string>
#include <sstream>

using namespace std;

size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

int main(int argc, const char *argv[]) {
    CURL *handle;
    CURLcode res;
    FILE *fp;

    /* URLs to download */
    string urlStart = "http://www3.cs.stonybrook.edu/~rezaul/"
                      "Fall-2016/CSE548/CSE548-lecture-";
    string urlEnd = ".pdf";

    /* No. of lectures */
    int lectures = 13;

    /* Out Location */
    string outLoc = argv[1];

    /* Set up CURL */
    curl_global_init(CURL_GLOBAL_ALL);
    handle = curl_easy_init();

    /* Do the download */
    if (handle) {
        for (int i = 1; i <= lectures; i++) {
            stringstream url;
            url << urlStart << to_string(i) << urlEnd;
            
            stringstream filename;
            filename << outLoc << "/CSE548-Lecture-" << to_string(i) << ".pdf";
            
            fp = fopen(filename.str().c_str(), "wb");
            curl_easy_setopt(handle, CURLOPT_URL, url.str().c_str());
            curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, write_data);
            curl_easy_setopt(handle, CURLOPT_WRITEDATA, fp);
            res = curl_easy_perform(handle);
            fclose(fp);

            /* Reuse handle if not on last pass */
            if (i != lectures) {
                curl_easy_reset(handle);
            }
        }
    } else {
        cout << "Unable to intialize CURL!";
    }

    curl_easy_cleanup(handle);
    curl_global_cleanup();

    return 0;
}